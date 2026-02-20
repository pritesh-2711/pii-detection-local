"""
Loads the consolidated PII dataset, applies:
  - finer_139 source cap (150k records)
  - rare entity type dropping (< 500 B- mentions -> collapsed to O)
  - stratified 80/10/10 split by source

Outputs:
  data/train.jsonl
  data/val.jsonl
  data/test.jsonl
  data/label_mapping.json
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

CONSOLIDATED_FILE = Path("./notebooks/pii_datasets/consolidated/consolidated.jsonl")
OUTPUT_DIR = Path("./data")
FINER_CAP = 150_000
RARE_THRESHOLD = 500
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# test gets the remainder (0.1)
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_consolidated(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records):,} records from {path}")
    return records


# ---------------------------------------------------------------------------
# Finer-139 cap
# ---------------------------------------------------------------------------

def cap_finer(records: list, cap: int, seed: int) -> list:
    finer = [r for r in records if r["source"] == "finer_139"]
    rest = [r for r in records if r["source"] != "finer_139"]
    rng = random.Random(seed)
    if len(finer) > cap:
        finer = rng.sample(finer, cap)
        print(f"finer_139 capped: {len(finer):,} records kept (was {len(records) - len(rest):,})")
    else:
        print(f"finer_139 under cap: {len(finer):,} records (no capping needed)")
    return rest + finer


# ---------------------------------------------------------------------------
# Rare type dropping
# ---------------------------------------------------------------------------

def drop_rare_entities(records: list, threshold: int) -> tuple:
    """
    Count B- mentions per entity type globally.
    Any type below threshold has all its B-/I- labels replaced with O.
    Returns (updated_records, kept_types, dropped_types).
    """
    mention_counts = Counter()
    for rec in records:
        for lbl in rec["labels"]:
            if lbl.startswith("B-"):
                mention_counts[lbl[2:]] += 1

    dropped_types = {t for t, c in mention_counts.items() if c < threshold}
    kept_types = {t for t, c in mention_counts.items() if c >= threshold}

    print("\nEntity type mention counts (before dropping):")
    for etype, count in sorted(mention_counts.items(), key=lambda x: -x[1]):
        status = "KEEP" if count >= threshold else "DROP"
        print(f"  [{status}] {etype:<35} {count:>10,}")

    if dropped_types:
        print(f"\nDropping {len(dropped_types)} rare entity type(s): {sorted(dropped_types)}")
        updated = []
        for rec in records:
            new_labels = []
            for lbl in rec["labels"]:
                if lbl == "O":
                    new_labels.append("O")
                elif lbl.startswith("B-") or lbl.startswith("I-"):
                    etype = lbl[2:]
                    new_labels.append("O" if etype in dropped_types else lbl)
                else:
                    new_labels.append("O")
            updated.append({
                "tokens": rec["tokens"],
                "labels": new_labels,
                "source": rec["source"],
            })
        records = updated
    else:
        print("\nNo rare entity types to drop.")

    return records, sorted(kept_types), sorted(dropped_types)


# ---------------------------------------------------------------------------
# Stratified split by source
# ---------------------------------------------------------------------------

def stratified_split(records: list, train_ratio: float, val_ratio: float,
                     seed: int) -> tuple:
    """
    For each source independently: shuffle, split 80/10/10.
    Concatenate across sources.
    """
    rng = random.Random(seed)
    by_source = defaultdict(list)
    for rec in records:
        by_source[rec["source"]].append(rec)

    train, val, test = [], [], []
    print("\nStratified split by source:")
    print(f"  {'Source':<30} {'Total':>8} {'Train':>8} {'Val':>8} {'Test':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for source, recs in sorted(by_source.items()):
        rng.shuffle(recs)
        n = len(recs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        src_train = recs[:n_train]
        src_val = recs[n_train:n_train + n_val]
        src_test = recs[n_train + n_val:]
        train.extend(src_train)
        val.extend(src_val)
        test.extend(src_test)
        print(f"  {source:<30} {n:>8,} {len(src_train):>8,} {len(src_val):>8,} {len(src_test):>8,}")

    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'TOTAL':<30} {len(train)+len(val)+len(test):>8,} {len(train):>8,} {len(val):>8,} {len(test):>8,}")

    # Final shuffle of each split so sources are interleaved during training
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def build_label_mapping(kept_types: list) -> tuple:
    labels = ["O"]
    for etype in sorted(kept_types):
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")
    label2id = {lbl: idx for idx, lbl in enumerate(labels)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}
    return labels, label2id, id2label


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_split(records: list, path: Path):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    size_mb = path.stat().st_size / 1e6
    print(f"  Saved {len(records):,} records -> {path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare():
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PII DATA PREPARATION")
    print("=" * 60)

    # 1. Load
    records = load_consolidated(CONSOLIDATED_FILE)

    # 2. Cap finer_139
    print(f"\nCapping finer_139 to {FINER_CAP:,} records ...")
    records = cap_finer(records, FINER_CAP, RANDOM_SEED)
    print(f"Total after cap: {len(records):,}")

    # 3. Drop rare entity types
    records, kept_types, dropped_types = drop_rare_entities(records, RARE_THRESHOLD)

    # 4. Build label mapping
    labels, label2id, id2label = build_label_mapping(kept_types)
    print(f"\nFinal label set ({len(labels)} labels including O):")
    for lbl in labels:
        print(f"  {lbl}")

    # 5. Stratified split
    train, val, test = stratified_split(records, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)

    # 6. Save splits
    print("\nSaving splits ...")
    save_split(train, OUTPUT_DIR / "train.jsonl")
    save_split(val, OUTPUT_DIR / "val.jsonl")
    save_split(test, OUTPUT_DIR / "test.jsonl")

    # 7. Save label mapping
    mapping = {
        "labels": labels,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "kept_entity_types": kept_types,
        "dropped_entity_types": dropped_types,
        "num_labels": len(labels),
    }
    label_path = OUTPUT_DIR / "label_mapping.json"
    with open(label_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"  Label mapping -> {label_path}")

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Entity types kept   : {len(kept_types)}")
    print(f"  Entity types dropped: {len(dropped_types)}")
    print(f"  Total labels        : {len(labels)}")
    print(f"  Train records       : {len(train):,}")
    print(f"  Val records         : {len(val):,}")
    print(f"  Test records        : {len(test):,}")

    return mapping


if __name__ == "__main__":
    prepare()
