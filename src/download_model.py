"""
Downloads microsoft/deberta-v3-base from HuggingFace Hub and saves all
required files to ./models/deberta-v3-base/.

Run once before training:
    python download_model.py

Requirements: transformers >= 4.30.0
"""

from pathlib import Path
from transformers import AutoTokenizer, DebertaV2Config

MODEL_ID = "microsoft/deberta-v3-base"
SAVE_DIR = Path("./models/deberta-v3-base")

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_ID} -> {SAVE_DIR.resolve()}")

    print("  Downloading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"  Tokenizer saved. Files: {[f.name for f in SAVE_DIR.iterdir()]}")

    print("  Downloading model weights (this may take a few minutes) ...")
    # Download base config only — no classification head yet.
    # train.py will add the classification head when it loads from this directory.
    config = DebertaV2Config.from_pretrained(MODEL_ID)
    config.save_pretrained(SAVE_DIR)

    from transformers import DebertaV2Model
    model = DebertaV2Model.from_pretrained(MODEL_ID)
    model.save_pretrained(SAVE_DIR)

    print("\nDone. Saved files:")
    for f in sorted(SAVE_DIR.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:<45} {size_mb:>8.1f} MB")

    total_mb = sum(f.stat().st_size for f in SAVE_DIR.iterdir()) / 1e6
    print(f"\n  Total: {total_mb:.1f} MB")
    print(f"\nModel ready at: {SAVE_DIR.resolve()}")
    print("You can now run the training pipeline.")

if __name__ == "__main__":
    main()
