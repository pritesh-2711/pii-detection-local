"""
Fine-tunes microsoft/deberta-v3-base for PII named entity recognition.
Configuration targets benchmark-level performance comparable to or exceeding
spaCy en_core_web_trf and Microsoft Presidio on PII detection tasks.

Cloud A100 (80GB):
    python src/train.py

Local GPU (<=8GB VRAM):
    PYTORCH_ALLOC_CONF=expandable_segments:True python src/train.py \
        --batch-size 2 --grad-accum 32 --gradient-checkpointing
"""

import json
import argparse
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR          = Path("./data")
MODELS_DIR        = Path("./models")
LOCAL_MODEL_PATH  = Path("./models/deberta-v3-base")
HF_MODEL_ID       = "microsoft/deberta-v3-base"

SEED              = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PIIDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, label2id: dict, max_length: int):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        self.tokenizer  = tokenizer
        self.label2id   = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec    = self.samples[idx]
        tokens = rec["tokens"]
        labels = rec["labels"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

        word_ids       = encoding.word_ids()
        aligned_labels = []
        prev_word_idx  = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != prev_word_idx:
                raw = labels[word_idx] if word_idx < len(labels) else "O"
                aligned_labels.append(self.label2id.get(raw, self.label2id["O"]))
            else:
                aligned_labels.append(-100)
            prev_word_idx = word_idx

        encoding["labels"] = aligned_labels
        return {k: torch.tensor(v) for k, v in encoding.items()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_compute_metrics(id2label: dict):
    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_labels, true_preds = [], []
        for pred_seq, label_seq in zip(predictions, label_ids):
            seq_labels, seq_preds = [], []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                seq_labels.append(id2label[int(l)])
                seq_preds.append(id2label[int(p)])
            true_labels.append(seq_labels)
            true_preds.append(seq_preds)

        return {
            "f1":        f1_score(true_labels, true_preds),
            "precision": precision_score(true_labels, true_preds),
            "recall":    recall_score(true_labels, true_preds),
        }
    return compute_metrics


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PIITrainer:
    def __init__(
        self,
        batch_size: int                  = 16,
        eval_batch_size: int             = 32,
        grad_accum: int                  = 4,
        learning_rate: float             = 2e-5,
        num_epochs: int                  = 10,
        max_steps: int                   = -1,
        max_length: int                  = 512,
        warmup_ratio: float              = 0.06,
        weight_decay: float              = 0.01,
        early_stopping_patience: int     = 3,
        eval_steps: int                  = 2000,
        save_steps: int                  = 2000,
        logging_steps: int               = 50,
        use_gradient_checkpointing: bool = False,
        resume_from_checkpoint: bool     = False,
    ):
        self.batch_size                  = batch_size
        self.eval_batch_size             = eval_batch_size
        self.grad_accum                  = grad_accum
        self.learning_rate               = learning_rate
        self.num_epochs                  = num_epochs
        self.max_steps                   = max_steps
        self.max_length                  = max_length
        self.warmup_ratio                = warmup_ratio
        self.weight_decay                = weight_decay
        self.early_stopping_patience     = early_stopping_patience
        self.eval_steps                  = eval_steps
        self.save_steps                  = save_steps
        self.logging_steps               = logging_steps
        self.use_gradient_checkpointing  = use_gradient_checkpointing
        self.resume_from_checkpoint      = resume_from_checkpoint

        # Label mapping
        with open(DATA_DIR / "label_mapping.json") as f:
            mapping = json.load(f)
        self.label2id   = mapping["label2id"]
        self.id2label   = {int(k): v for k, v in mapping["id2label"].items()}
        self.labels     = mapping["labels"]
        self.num_labels = mapping["num_labels"]

        print(f"Labels      : {self.num_labels} total ({len(self.labels)} including O)")

        # Model + tokenizer
        model_source = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else HF_MODEL_ID
        print(f"Model source: {model_source}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_source)

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_source,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        print(f"Device      : {self.device}")
        print(f"Precision   : {'bf16' if self.use_bf16 else 'fp32'}")
        print(f"Grad ckpt   : {'enabled' if self.use_gradient_checkpointing else 'disabled'}")
        print(f"Max length  : {self.max_length}")
        if self.max_steps > 0:
            print(f"Max steps   : {self.max_steps} (overrides epochs)")

    def load_datasets(self):
        print("\nLoading datasets ...")
        self.train_ds = PIIDataset(DATA_DIR / "train.jsonl", self.tokenizer, self.label2id, self.max_length)
        self.val_ds   = PIIDataset(DATA_DIR / "val.jsonl",   self.tokenizer, self.label2id, self.max_length)
        self.test_ds  = PIIDataset(DATA_DIR / "test.jsonl",  self.tokenizer, self.label2id, self.max_length)
        print(f"  Train : {len(self.train_ds):,}")
        print(f"  Val   : {len(self.val_ds):,}")
        print(f"  Test  : {len(self.test_ds):,}")

    def train(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = MODELS_DIR / "checkpoints"

        effective_batch  = self.batch_size * self.grad_accum
        steps_per_epoch  = max(1, len(self.train_ds) // effective_batch)

        # When max_steps is set, use it for warmup calculation.
        # Otherwise use total epoch-based steps.
        if self.max_steps > 0:
            total_steps_for_warmup = self.max_steps
        else:
            total_steps_for_warmup = steps_per_epoch * self.num_epochs
        warmup_steps = int(total_steps_for_warmup * self.warmup_ratio)

        print(f"\n  Effective batch size  : {effective_batch}")
        print(f"  Steps per epoch       : {steps_per_epoch:,}")
        print(f"  Total optimizer steps : {total_steps_for_warmup:,}")
        print(f"  Warmup steps          : {warmup_steps:,}")
        print(f"  Early stop patience   : {self.early_stopping_patience} epochs")
        print(f"  Eval every            : {self.eval_steps} steps")
        print(f"  Save every            : {self.save_steps} steps")
        print(f"  Log every             : {self.logging_steps} steps")

        on_cloud    = not self.use_gradient_checkpointing
        pin_memory  = on_cloud
        num_workers = 2  # 4 causes CPU contention on g2-standard-8

        args = TrainingArguments(
            output_dir=str(output_dir),
            seed=SEED,

            # Core training
            num_train_epochs=self.num_epochs,
            max_steps=self.max_steps,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            gradient_accumulation_steps=self.grad_accum,

            # Optimiser
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=warmup_steps,
            lr_scheduler_type="linear",
            max_grad_norm=1.0,

            # Precision
            fp16=False,
            bf16=self.use_bf16,

            # Evaluation & checkpointing — steps-based
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=3,

            # Logging
            logging_strategy="steps",
            logging_steps=self.logging_steps,
            report_to="none",

            # Dataloader
            dataloader_pin_memory=pin_memory,
            dataloader_num_workers=num_workers,

            # Memory
            gradient_checkpointing=self.use_gradient_checkpointing,
        )

        collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.max_length,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_ds,
            eval_dataset=self.val_ds,
            data_collator=collator,
            processing_class=self.tokenizer,
            compute_metrics=make_compute_metrics(self.id2label),
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.early_stopping_patience
                )
            ],
        )

        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)

        checkpoint = None
        if self.resume_from_checkpoint:
            checkpoints_dir = MODELS_DIR / "checkpoints"
            if checkpoints_dir.exists():
                ckpts = sorted(checkpoints_dir.glob("checkpoint-*"),
                               key=lambda p: int(p.name.split("-")[-1]))
                if ckpts:
                    checkpoint = str(ckpts[-1])
                    print(f"Resuming from checkpoint: {checkpoint}")
                else:
                    print("No checkpoint found, starting from scratch.")
            else:
                print("No checkpoints directory found, starting from scratch.")

        trainer.train(resume_from_checkpoint=checkpoint)

        best_model_dir = MODELS_DIR / "best_model"
        print(f"\nSaving best model to {best_model_dir} ...")
        trainer.save_model(str(best_model_dir))
        self.tokenizer.save_pretrained(str(best_model_dir))
        with open(best_model_dir / "label_mapping.json", "w") as f:
            json.dump({
                "labels":     self.labels,
                "label2id":   self.label2id,
                "id2label":   {str(k): v for k, v in self.id2label.items()},
                "num_labels": self.num_labels,
            }, f, indent=2)

        return trainer

    def evaluate(self, trainer):
        print("\n" + "=" * 60)
        print("EVALUATION ON TEST SET")
        print("=" * 60)

        preds_output = trainer.predict(self.test_ds, metric_key_prefix="test")
        logits       = preds_output.predictions
        label_ids    = preds_output.label_ids
        predictions  = np.argmax(logits, axis=-1)

        true_labels, true_preds = [], []
        for pred_seq, label_seq in zip(predictions, label_ids):
            seq_labels, seq_preds = [], []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                seq_labels.append(self.id2label[int(l)])
                seq_preds.append(self.id2label[int(p)])
            true_labels.append(seq_labels)
            true_preds.append(seq_preds)

        print("\nPer-entity F1 (seqeval span-level):")
        print(classification_report(true_labels, true_preds, digits=4))

        results = {
            "test_f1":        f1_score(true_labels, true_preds),
            "test_precision": precision_score(true_labels, true_preds),
            "test_recall":    recall_score(true_labels, true_preds),
        }
        print(f"Overall F1        : {results['test_f1']:.4f}")
        print(f"Overall Precision : {results['test_precision']:.4f}")
        print(f"Overall Recall    : {results['test_recall']:.4f}")

        results_path = MODELS_DIR / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to  : {results_path}")

        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa-v3-base for PII NER")

    parser.add_argument("--batch-size",              type=int,   default=16)
    parser.add_argument("--eval-batch-size",          type=int,   default=32,
                        help="Eval/predict batch size (default: 32). Lower if eval OOMs.")
    parser.add_argument("--grad-accum",              type=int,   default=4)
    parser.add_argument("--epochs",                  type=int,   default=10)
    parser.add_argument("--max-steps",               type=int,   default=-1,
                        help="Hard step limit. Overrides --epochs when > 0.")
    parser.add_argument("--max-length",              type=int,   default=512,
                        help="Token sequence length (default: 512)")
    parser.add_argument("--lr",                      type=float, default=2e-5)
    parser.add_argument("--warmup-ratio",            type=float, default=0.06)
    parser.add_argument("--weight-decay",            type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int,   default=3)
    parser.add_argument("--eval-steps",              type=int,   default=2000,
                        help="Evaluate every N steps (default: 2000)")
    parser.add_argument("--save-steps",              type=int,   default=2000,
                        help="Save checkpoint every N steps (default: 2000)")
    parser.add_argument("--logging-steps",           type=int,   default=50,
                        help="Log loss/lr every N steps (default: 50)")
    parser.add_argument("--gradient-checkpointing",  action="store_true")
    parser.add_argument("--resume-from-checkpoint",  action="store_true",
                        help="Resume from latest checkpoint in models/checkpoints/")

    args = parser.parse_args()

    pii_trainer = PIITrainer(
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        use_gradient_checkpointing=args.gradient_checkpointing,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    pii_trainer.load_datasets()
    trainer = pii_trainer.train()
    pii_trainer.evaluate(trainer)


if __name__ == "__main__":
    main()