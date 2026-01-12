import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

class PIIDataset(Dataset):
    """
    Dataset class for PII token classification.
    """
    
    def __init__(self, data_path: str, tokenizer, label2id: Dict[str, int], max_length: int = 128):
        self.data = pd.read_json(data_path, lines=True)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = row['tokens']
        labels = row['labels']
        
        # Tokenize with alignment
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Align labels with subword tokens
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 (ignored in loss)
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First subword token of a word gets the label
                label = labels[word_idx] if word_idx < len(labels) else "O"
                aligned_labels.append(self.label2id.get(label, self.label2id["O"]))
            else:
                # Subsequent subword tokens get -100 or same label
                aligned_labels.append(-100)
            
            previous_word_idx = word_idx
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

class PIITrainer:
    """
    Trainer for PII detection model.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-cased",
        data_dir: str = "./data",
        output_dir: str = "./models",
        max_length: int = 128,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        num_epochs: int = 5
    ):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load label mappings
        with open(self.data_dir / "label_mapping.json", 'r') as f:
            label_info = json.load(f)
        
        self.label2id = label_info['label2id']
        self.id2label = {int(k): v for k, v in label_info['id2label'].items()}
        self.labels = label_info['labels']
        
        print(f"Loaded {len(self.labels)} labels")
        
        # Initialize tokenizer and model
        print(f"\nLoading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Check for GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def load_datasets(self):
        """
        Load train, validation, and test datasets.
        """
        print("\nLoading datasets...")
        
        self.train_dataset = PIIDataset(
            self.data_dir / "train.jsonl",
            self.tokenizer,
            self.label2id,
            self.max_length
        )
        
        self.val_dataset = PIIDataset(
            self.data_dir / "val.jsonl",
            self.tokenizer,
            self.label2id,
            self.max_length
        )
        
        self.test_dataset = PIIDataset(
            self.data_dir / "test.jsonl",
            self.tokenizer,
            self.label2id,
            self.max_length
        )
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
    
    def compute_metrics(self, eval_pred):
        """
        Compute seqeval metrics for NER evaluation.
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_labels = []
        true_predictions = []
        
        for prediction, label in zip(predictions, labels):
            true_label = []
            true_prediction = []
            
            for pred_id, label_id in zip(prediction, label):
                if label_id != -100:
                    true_label.append(self.id2label[label_id])
                    true_prediction.append(self.id2label[pred_id])
            
            true_labels.append(true_label)
            true_predictions.append(true_prediction)
        
        # Calculate metrics
        results = {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
        
        return results
    
    def train(self):
        """
        Fine-tune the model.
        """
        print("\n=== Starting Training ===")
        
        # Training arguments - FIXED: eval_strategy instead of evaluation_strategy
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=100,
            warmup_steps=500,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        trainer.train()
        
        # Save best model
        print("\nSaving final model...")
        trainer.save_model(self.output_dir / "best_model")
        self.tokenizer.save_pretrained(self.output_dir / "best_model")
        
        # Save label mappings with model
        with open(self.output_dir / "best_model" / "label_mapping.json", 'w') as f:
            json.dump({
                'label2id': self.label2id,
                'id2label': self.id2label,
                'labels': self.labels
            }, f, indent=2)
        
        print(f"Model saved to {self.output_dir / 'best_model'}")
        
        return trainer
    
    def evaluate(self, trainer):
        """
        Evaluate on test set.
        """
        print("\n=== Evaluating on Test Set ===")
        
        results = trainer.evaluate(self.test_dataset)
        
        print("\nTest Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
        
        # Detailed classification report
        predictions = trainer.predict(self.test_dataset)
        preds = np.argmax(predictions.predictions, axis=2)
        labels = predictions.label_ids
        
        true_labels = []
        true_predictions = []
        
        for prediction, label in zip(preds, labels):
            true_label = []
            true_prediction = []
            
            for pred_id, label_id in zip(prediction, label):
                if label_id != -100:
                    true_label.append(self.id2label[label_id])
                    true_prediction.append(self.id2label[pred_id])
            
            true_labels.append(true_label)
            true_predictions.append(true_prediction)
        
        print("\nDetailed Classification Report:")
        print(classification_report(true_labels, true_predictions))
        
        # Save evaluation results
        with open(self.output_dir / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def main():
    """
    Main training pipeline.
    """
    # Configuration
    config = {
        'model_name': 'bert-base-cased',
        'data_dir': './data',
        'output_dir': './models',
        'max_length': 128,
        'batch_size': 16,
        'learning_rate': 5e-5,
        'num_epochs': 5
    }
    
    print("=== PII Detection Model Training ===")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Initialize trainer
    pii_trainer = PIITrainer(**config)
    
    # Load datasets
    pii_trainer.load_datasets()
    
    # Train model
    trainer = pii_trainer.train()
    
    # Evaluate
    pii_trainer.evaluate(trainer)
    
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()