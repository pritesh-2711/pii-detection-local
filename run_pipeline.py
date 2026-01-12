#!/usr/bin/env python3
"""
Complete pipeline for PII detection model.
Runs data preparation, training, and evaluation.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_preparation import PIIDataCollector
from train import PIITrainer

def run_pipeline(
    data_dir: str = "./data",
    models_dir: str = "./models",
    model_name: str = "bert-base-cased",
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    skip_data_prep: bool = False
):
    """
    Run complete training pipeline.
    """
    print("="*80)
    print("PII DETECTION MODEL - TRAINING PIPELINE")
    print("="*80)
    
    # Step 1: Data Preparation
    if not skip_data_prep:
        print("\n[STEP 1/3] Data Preparation")
        print("-"*80)
        collector = PIIDataCollector(data_dir=data_dir)
        datasets = collector.prepare_dataset()
        print(f"\nData preparation complete. Datasets saved to {data_dir}")
    else:
        print("\n[STEP 1/3] Skipping data preparation (using existing data)")
    
    # Step 2: Model Training
    print("\n[STEP 2/3] Model Training")
    print("-"*80)
    
    trainer = PIITrainer(
        model_name=model_name,
        data_dir=data_dir,
        output_dir=models_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )
    
    trainer.load_datasets()
    trained_model = trainer.train()
    
    # Step 3: Evaluation
    print("\n[STEP 3/3] Model Evaluation")
    print("-"*80)
    
    results = trainer.evaluate(trained_model)
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nModel saved to: {models_dir}/best_model")
    print(f"Test F1 Score: {results.get('eval_f1', 0):.4f}")
    print(f"Test Precision: {results.get('eval_precision', 0):.4f}")
    print(f"Test Recall: {results.get('eval_recall', 0):.4f}")
    print("\nNext steps:")
    print("  1. Test the model: python src/inference.py")
    print("  2. Start API server: python src/api.py --model-path ./models/best_model")
    print("  3. Deploy with Docker: docker build -t pii-detector .")

def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline for PII detection model training"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for dataset storage"
    )
    
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory for model storage"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-cased",
        choices=["bert-base-cased", "roberta-base", "microsoft/deberta-v3-base"],
        help="Base model to fine-tune"
    )
    
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip data preparation (use existing data)"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        skip_data_prep=args.skip_data_prep
    )

if __name__ == "__main__":
    main()
