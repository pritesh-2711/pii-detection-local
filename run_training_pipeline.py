"""
End-to-end training pipeline for PII detection model.

Steps:
    1. Download microsoft/deberta-v3-base weights -> models/deberta-v3-base/
    2. Fine-tune on PII NER data -> models/best_model/

Usage:
    python run_training_pipeline.py

    # Skip model download if already done:
    python run_training_pipeline.py --skip-download

    # Custom training hyperparameters:
    python run_training_pipeline.py \
        --batch-size 2 \
        --grad-accum 32 \
        --epochs 10 \
        --lr 2e-5 \
        --gradient-checkpointing

    # Full options:
    python run_training_pipeline.py --help
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from download_model import main as download_model
from train import PIITrainer


def run_pipeline(
    skip_download: bool = False,
    batch_size: int = 16,
    grad_accum: int = 4,
    learning_rate: float = 2e-5,
    num_epochs: int = 10,
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
    early_stopping_patience: int = 3,
    use_gradient_checkpointing: bool = False,
):
    print("=" * 80)
    print("PII DETECTION — TRAINING PIPELINE")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Step 1: Download base model
    # ------------------------------------------------------------------
    if not skip_download:
        print("\n[STEP 1/2] Downloading microsoft/deberta-v3-base")
        print("-" * 80)
        download_model()
        print("\nModel download complete.")
    else:
        print("\n[STEP 1/2] Skipping model download (using existing weights in "
              "./models/deberta-v3-base)")

    # ------------------------------------------------------------------
    # Step 2: Fine-tune
    # ------------------------------------------------------------------
    print("\n[STEP 2/2] Fine-tuning on PII NER data")
    print("-" * 80)

    trainer = PIITrainer(
        batch_size=batch_size,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    trainer.load_datasets()
    hf_trainer = trainer.train()
    results = trainer.evaluate(hf_trainer)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 80)
    print(f"  Best model        : ./models/best_model")
    print(f"  Evaluation results: ./models/evaluation_results.json")
    print(f"  Test F1           : {results.get('test_f1', 0):.4f}")
    print(f"  Test Precision    : {results.get('test_precision', 0):.4f}")
    print(f"  Test Recall       : {results.get('test_recall', 0):.4f}")
    print("\nNext step: python src/api.py --model-path ./models/best_model")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end training pipeline for PII detection model"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip base model download and use existing weights in ./models/deberta-v3-base",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-device train batch size (default: 16)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.06,
        help="Warmup ratio as fraction of total steps (default: 0.06)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Stop if val F1 does not improve for N epochs (default: 3)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing — required on <=8GB VRAM",
    )

    args = parser.parse_args()

    run_pipeline(
        skip_download=args.skip_download,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )


if __name__ == "__main__":
    main()