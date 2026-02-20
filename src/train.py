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

MAX_LENGTH        = 512
SEED              = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PIIDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, label2id: dict):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.label2id  = label2id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec    = self.samples[idx]
        tokens = rec["tokens"]
        labels = rec["labels"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=MAX_LENGTH,
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
                # Continuation subword — excluded from loss
                aligned_labels.append(-100)
            prev_word_idx = word_idx

        encoding["labels"] = aligned_labels
        return {k: torch.tensor(v) for k, v in encoding.items()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_compute_metrics(id2label: dict):
    """
    Returns seqeval-based compute_metrics function.

    Reports:
        f1        — micro-averaged over all entity types (primary benchmark metric)
        precision — micro-averaged
        recall    — micro-averaged

    seqeval uses span-level evaluation (not token-level), which is the standard
    used by CoNLL-2003, OntoNotes, and all major NER benchmarks. This makes
    results directly comparable to published DeBERTa-v3 NER papers.
    """
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
        grad_accum: int                  = 4,
        learning_rate: float             = 2e-5,
        num_epochs: int                  = 10,
        warmup_ratio: float              = 0.06,
        weight_decay: float              = 0.01,
        early_stopping_patience: int     = 3,
        use_gradient_checkpointing: bool = False,
    ):
        self.batch_size                  = batch_size
        self.grad_accum                  = grad_accum
        self.learning_rate               = learning_rate
        self.num_epochs                  = num_epochs
        self.warmup_ratio                = warmup_ratio
        self.weight_decay                = weight_decay
        self.early_stopping_patience     = early_stopping_patience
        self.use_gradient_checkpointing  = use_gradient_checkpointing

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
            ignore_mismatched_sizes=True,   # base checkpoint has no cls head
        )

        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # DeBERTa-v3 disentangled attention produces gradients that overflow fp16
        # during GradScaler unscale. bf16 has the same exponent range as fp32
        # and avoids this. Fall back to fp32 on pre-Ampere GPUs.
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        print(f"Device      : {self.device}")
        print(f"Precision   : {'bf16' if self.use_bf16 else 'fp32'}")
        print(f"Grad ckpt   : {'enabled' if self.use_gradient_checkpointing else 'disabled'}")

    def load_datasets(self):
        print("\nLoading datasets ...")
        self.train_ds = PIIDataset(DATA_DIR / "train.jsonl", self.tokenizer, self.label2id)
        self.val_ds   = PIIDataset(DATA_DIR / "val.jsonl",   self.tokenizer, self.label2id)
        self.test_ds  = PIIDataset(DATA_DIR / "test.jsonl",  self.tokenizer, self.label2id)
        print(f"  Train : {len(self.train_ds):,}")
        print(f"  Val   : {len(self.val_ds):,}")
        print(f"  Test  : {len(self.test_ds):,}")

    def train(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = MODELS_DIR / "checkpoints"

        effective_batch  = self.batch_size * self.grad_accum
        steps_per_epoch  = max(1, len(self.train_ds) // effective_batch)
        total_steps      = steps_per_epoch * self.num_epochs
        warmup_steps     = int(total_steps * self.warmup_ratio)

        print(f"\n  Effective batch size  : {effective_batch}")
        print(f"  Steps per epoch       : {steps_per_epoch:,}")
        print(f"  Total optimizer steps : {total_steps:,}")
        print(f"  Warmup steps          : {warmup_steps:,}")
        print(f"  Early stop patience   : {self.early_stopping_patience} epochs")

        # dataloader tuning: pin_memory + more workers benefit cloud NVMe setups
        # disable on local machines where GPU memory is the bottleneck
        on_cloud    = not self.use_gradient_checkpointing
        pin_memory  = on_cloud
        num_workers = 4 if on_cloud else 2

        args = TrainingArguments(
            output_dir=str(output_dir),
            seed=SEED,

            # Core training
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum,

            # Optimiser — AdamW with linear LR decay to zero
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=warmup_steps,
            lr_scheduler_type="linear",     # standard for NER fine-tuning
            max_grad_norm=1.0,              # gradient clipping, DeBERTa default

            # Precision
            fp16=False,
            bf16=self.use_bf16,

            # Evaluation & checkpointing
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=3,             # keep best + 2 most recent

            # Logging
            logging_strategy="steps",
            logging_steps=500,
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
            max_length=MAX_LENGTH,
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
        trainer.train()

        # Persist best model + tokenizer + label mapping together
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

        preds_output = trainer.predict(self.test_ds)
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

    parser.add_argument("--batch-size",             type=int,   default=16,
                        help="Per-device train batch size (default: 16 for A100)")
    parser.add_argument("--grad-accum",             type=int,   default=4,
                        help="Gradient accumulation steps (effective batch = batch * accum)")
    parser.add_argument("--epochs",                 type=int,   default=10)
    parser.add_argument("--lr",                     type=float, default=2e-5)
    parser.add_argument("--warmup-ratio",           type=float, default=0.06)
    parser.add_argument("--weight-decay",           type=float, default=0.01)
    parser.add_argument("--early-stopping-patience",type=int,   default=3,
                        help="Stop if val F1 does not improve for N epochs")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing (required on <=8GB VRAM)")

    args = parser.parse_args()

    pii_trainer = PIITrainer(
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )
    pii_trainer.load_datasets()
    trainer = pii_trainer.train()
    pii_trainer.evaluate(trainer)


if __name__ == "__main__":
    main()