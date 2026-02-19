"""
PII Dataset Consolidation Script
=================================
Reads all downloaded JSONL files, normalises every source into a unified
BIO-tagged format, reports unique entity types across all sources, and
writes a single consolidated JSONL file ready for transformer fine-tuning.

Output schema (one JSON object per line):
  {
    "tokens":  ["Pritesh", "works", "at", "XYZ-Corp"],
    "labels":  ["B-PERSON", "O", "O", "B-ORG"],
    "source":  "conll2003"
  }

Usage:
  python consolidate_pii_datasets.py \
      --data-dir ./pii_datasets \
      --output-dir ./pii_datasets/consolidated

Example:
    python notebooks/consolidate_pii_datasets.py \
        --data-dir notebooks/pii_datasets \
        --output-dir notebooks/pii_datasets/consolidated
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Entity label normalisation map
# ---------------------------------------------------------------------------
# Every source uses different label names for the same concept.
# This maps all variants to a single canonical label set.
LABEL_NORM = {
    # Person
    "PER": "PERSON", "B-PER": "B-PERSON", "I-PER": "I-PERSON",
    "FIRSTNAME": "PERSON", "B-FIRSTNAME": "B-PERSON", "I-FIRSTNAME": "I-PERSON",
    "LASTNAME": "PERSON", "B-LASTNAME": "B-PERSON", "I-LASTNAME": "I-PERSON",
    "MIDDLENAME": "PERSON", "B-MIDDLENAME": "B-PERSON", "I-MIDDLENAME": "I-PERSON",
    "PREFIX": "PERSON", "B-PREFIX": "B-PERSON", "I-PREFIX": "I-PERSON",
    "GENDER": "PERSON", "B-GENDER": "B-PERSON", "I-GENDER": "I-PERSON",
    "SEX": "PERSON", "B-SEX": "B-PERSON", "I-SEX": "I-PERSON",
    "AGE": "PERSON", "B-AGE": "B-PERSON", "I-AGE": "I-PERSON",
    "DOB": "DATE", "B-DOB": "B-DATE", "I-DOB": "I-DATE",
    "EYECOLOR": "PERSON", "B-EYECOLOR": "B-PERSON", "I-EYECOLOR": "I-PERSON",
    "HEIGHT": "PERSON", "B-HEIGHT": "B-PERSON", "I-HEIGHT": "I-PERSON",

    # Organisation
    "ORG": "ORG",
    "COMPANYNAME": "ORG", "B-COMPANYNAME": "B-ORG", "I-COMPANYNAME": "I-ORG",
    "ACCOUNTNAME": "ORG", "B-ACCOUNTNAME": "B-ORG", "I-ACCOUNTNAME": "I-ORG",

    # Location
    "LOC": "LOC",
    "CITY": "LOC", "B-CITY": "B-LOC", "I-CITY": "I-LOC",
    "STATE": "LOC", "B-STATE": "B-LOC", "I-STATE": "I-LOC",
    "COUNTY": "LOC", "B-COUNTY": "B-LOC", "I-COUNTY": "I-LOC",
    "ZIPCODE": "LOC", "B-ZIPCODE": "B-LOC", "I-ZIPCODE": "I-LOC",
    "STREET": "ADDRESS", "B-STREET": "B-ADDRESS", "I-STREET": "I-ADDRESS",
    "BUILDINGNUMBER": "ADDRESS", "B-BUILDINGNUMBER": "B-ADDRESS", "I-BUILDINGNUMBER": "I-ADDRESS",
    "SECONDARYADDRESS": "ADDRESS", "B-SECONDARYADDRESS": "B-ADDRESS", "I-SECONDARYADDRESS": "I-ADDRESS",
    "NEARBYGPSCOORDINATE": "LOC", "B-NEARBYGPSCOORDINATE": "B-LOC", "I-NEARBYGPSCOORDINATE": "I-LOC",
    "ORDINALDIRECTION": "LOC", "B-ORDINALDIRECTION": "B-LOC", "I-ORDINALDIRECTION": "I-LOC",

    # Email
    "EMAIL": "EMAIL",

    # Phone
    "PHONENUMBER": "PHONE", "B-PHONENUMBER": "B-PHONE", "I-PHONENUMBER": "I-PHONE",
    "PHONE": "PHONE",
    "PHONEIMEI": "PHONE", "B-PHONEIMEI": "B-PHONE", "I-PHONEIMEI": "I-PHONE",

    # Financial identifiers
    "CREDITCARDNUMBER": "CREDIT_CARD", "B-CREDITCARDNUMBER": "B-CREDIT_CARD", "I-CREDITCARDNUMBER": "I-CREDIT_CARD",
    "CREDITCARDCVV": "CREDIT_CARD", "B-CREDITCARDCVV": "B-CREDIT_CARD", "I-CREDITCARDCVV": "I-CREDIT_CARD",
    "CREDITCARDISSUER": "CREDIT_CARD", "B-CREDITCARDISSUER": "B-CREDIT_CARD", "I-CREDITCARDISSUER": "I-CREDIT_CARD",
    "CREDIT_CARD": "CREDIT_CARD",
    "IBAN": "IBAN",
    "BIC": "BIC",
    "ACCOUNTNUMBER": "ACCOUNT_NUMBER", "B-ACCOUNTNUMBER": "B-ACCOUNT_NUMBER", "I-ACCOUNTNUMBER": "I-ACCOUNT_NUMBER",
    "ACCOUNT_NUMBER": "ACCOUNT_NUMBER",
    "ROUTING_NUMBER": "ROUTING_NUMBER",
    "MASKEDNUMBER": "ACCOUNT_NUMBER", "B-MASKEDNUMBER": "B-ACCOUNT_NUMBER", "I-MASKEDNUMBER": "I-ACCOUNT_NUMBER",
    "PIN": "PIN",
    "TAX_ID": "TAX_ID",
    "SSN": "SSN",

    # Currency / Amount
    "AMOUNT": "AMOUNT",
    "CURRENCY": "CURRENCY", "B-CURRENCY": "B-CURRENCY", "I-CURRENCY": "I-CURRENCY",
    "CURRENCYCODE": "CURRENCY", "B-CURRENCYCODE": "B-CURRENCY", "I-CURRENCYCODE": "I-CURRENCY",
    "CURRENCYNAME": "CURRENCY", "B-CURRENCYNAME": "B-CURRENCY", "I-CURRENCYNAME": "I-CURRENCY",
    "CURRENCYSYMBOL": "CURRENCY", "B-CURRENCYSYMBOL": "B-CURRENCY", "I-CURRENCYSYMBOL": "I-CURRENCY",

    # Crypto
    "BITCOINADDRESS": "CRYPTO_ADDRESS", "B-BITCOINADDRESS": "B-CRYPTO_ADDRESS", "I-BITCOINADDRESS": "I-CRYPTO_ADDRESS",
    "ETHEREUMADDRESS": "CRYPTO_ADDRESS", "B-ETHEREUMADDRESS": "B-CRYPTO_ADDRESS", "I-ETHEREUMADDRESS": "I-CRYPTO_ADDRESS",
    "LITECOINADDRESS": "CRYPTO_ADDRESS", "B-LITECOINADDRESS": "B-CRYPTO_ADDRESS", "I-LITECOINADDRESS": "I-CRYPTO_ADDRESS",

    # Network / Device
    "IP": "IP_ADDRESS", "B-IP": "B-IP_ADDRESS", "I-IP": "I-IP_ADDRESS",
    "IPV4": "IP_ADDRESS", "B-IPV4": "B-IP_ADDRESS", "I-IPV4": "I-IP_ADDRESS",
    "IPV6": "IP_ADDRESS", "B-IPV6": "B-IP_ADDRESS", "I-IPV6": "I-IP_ADDRESS",
    "MAC": "IP_ADDRESS", "B-MAC": "B-IP_ADDRESS", "I-MAC": "I-IP_ADDRESS",
    "USERAGENT": "USERNAME", "B-USERAGENT": "B-USERNAME", "I-USERAGENT": "I-USERNAME",
    "URL": "URL",

    # Auth
    "USERNAME": "USERNAME",
    "PASSWORD": "PASSWORD",

    # Date / Time
    "DATE": "DATE",
    "TIME": "TIME",

    # Job
    "JOBTITLE": "JOB", "B-JOBTITLE": "B-JOB", "I-JOBTITLE": "I-JOB",
    "JOBAREA": "JOB", "B-JOBAREA": "B-JOB", "I-JOBAREA": "I-JOB",
    "JOBTYPE": "JOB", "B-JOBTYPE": "B-JOB", "I-JOBTYPE": "I-JOB",

    # Vehicle
    "VEHICLEVIN": "VEHICLE", "B-VEHICLEVIN": "B-VEHICLE", "I-VEHICLEVIN": "I-VEHICLE",
    "VEHICLEVRM": "VEHICLE", "B-VEHICLEVRM": "B-VEHICLE", "I-VEHICLEVRM": "I-VEHICLE",
    "VEHI": "VEHICLE", "B-VEHI": "B-VEHICLE", "I-VEHI": "I-VEHICLE",

    # Other multinerd / few-nerd types kept as-is (no PII relevance, pass through)
    "MISC": "MISC",
    "ANIM": "MISC", "BIO": "MISC", "CEL": "MISC", "DIS": "MISC",
    "EVE": "MISC", "FOOD": "MISC", "INST": "MISC", "MEDIA": "MISC",
    "MYTH": "MISC", "PLANT": "MISC",

    # FiNER-139 financial numeric entities — kept as FINANCIAL_ENTITY
    # (these are XBRL tags like Revenue, Assets, etc.)
}

# Canonical entity types that are PII-relevant for banking
BANKING_PII_TYPES = {
    "PERSON", "ORG", "LOC", "ADDRESS", "EMAIL", "PHONE",
    "CREDIT_CARD", "IBAN", "BIC", "ACCOUNT_NUMBER", "ROUTING_NUMBER",
    "PIN", "TAX_ID", "SSN", "AMOUNT", "CURRENCY", "CRYPTO_ADDRESS",
    "IP_ADDRESS", "USERNAME", "PASSWORD", "URL", "DATE", "TIME",
    "JOB", "VEHICLE", "MISC",
}


def normalise_label(label: str) -> str:
    """
    Normalise a BIO label to the canonical label set.
    Handles B-/I- prefixes and unknown labels.
    """
    if label == "O":
        return "O"

    prefix = ""
    base = label
    if label.startswith("B-") or label.startswith("I-"):
        prefix = label[:2]
        base = label[2:]

    # Direct lookup with prefix
    full_key = prefix + base
    if full_key in LABEL_NORM:
        normed = LABEL_NORM[full_key]
        # If the norm result already has a prefix, return as-is
        if normed.startswith("B-") or normed.startswith("I-"):
            return normed
        return prefix + normed

    # Lookup base only
    if base in LABEL_NORM:
        normed = LABEL_NORM[base]
        if normed.startswith("B-") or normed.startswith("I-"):
            return normed
        return prefix + normed

    # Unknown — treat as FINANCIAL_ENTITY if it looks like an XBRL tag
    # (starts with uppercase, camelCase, no dash)
    if base and base[0].isupper() and "-" not in base:
        return prefix + "FINANCIAL_ENTITY"

    return prefix + base.upper()


# ---------------------------------------------------------------------------
# Span-to-BIO converter (for gretel and nvidia datasets)
# ---------------------------------------------------------------------------

def span_to_bio(text: str, spans: list) -> tuple[list, list]:
    """
    Convert a raw text + list of character-offset spans into
    whitespace-tokenised tokens and BIO labels.

    Each span is expected to be a dict with keys:
      - 'start' / 'begin': int  (char offset, inclusive)
      - 'end':             int  (char offset, exclusive)
      - 'type' / 'label' / 'entity_type': str
    """
    tokens = text.split()
    labels = ["O"] * len(tokens)

    # Build char-offset -> token index map
    char_to_tok = {}
    pos = 0
    for tok_idx, tok in enumerate(tokens):
        start = text.find(tok, pos)
        for c in range(start, start + len(tok)):
            char_to_tok[c] = tok_idx
        pos = start + len(tok)

    for span in spans:
        # Normalise span dict keys
        start = span.get("start", span.get("begin", span.get("char_start")))
        end = span.get("end", span.get("char_end"))
        label = span.get("type", span.get("label", span.get("entity_type", span.get("tag", ""))))

        if start is None or end is None or not label:
            continue

        first_tok = char_to_tok.get(start)
        last_tok = char_to_tok.get(end - 1)

        if first_tok is None or last_tok is None:
            continue

        labels[first_tok] = f"B-{label}"
        for i in range(first_tok + 1, last_tok + 1):
            labels[i] = f"I-{label}"

    return tokens, labels


def parse_span_field(raw) -> list:
    """
    Parse the pii_spans / spans column which may be a JSON string or a list.
    Returns a flat list of span dicts.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        # Could be list of dicts or list of lists
        result = []
        for item in raw:
            if isinstance(item, dict):
                result.append(item)
            elif isinstance(item, str):
                try:
                    result.append(json.loads(item))
                except Exception:
                    pass
        return result
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return []
    return []


# ---------------------------------------------------------------------------
# Per-source readers
# ---------------------------------------------------------------------------

def read_bio_jsonl(filepath: Path, token_col: str, label_col: str,
                   source: str, label_names: list = None) -> list:
    """
    Generic reader for datasets that already have tokens + BIO labels as lists.
    label_names: if labels are integers, this list maps id -> label string.
    """
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tokens = row.get(token_col)
            labels = row.get(label_col)
            if not tokens or not labels:
                continue

            # Decode integer labels
            if label_names and labels and isinstance(labels[0], int):
                labels = [label_names[i] if i < len(label_names) else "O" for i in labels]

            # Normalise
            labels = [normalise_label(str(l)) for l in labels]

            if len(tokens) != len(labels):
                # Truncate to shorter — shouldn't happen often
                min_len = min(len(tokens), len(labels))
                tokens = tokens[:min_len]
                labels = labels[:min_len]

            records.append({"tokens": tokens, "labels": labels, "source": source})
    return records


def read_span_jsonl(filepath: Path, text_col: str, span_col: str, source: str) -> list:
    """
    Reader for datasets with raw text + character-offset span annotations.
    """
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get(text_col, "")
            spans_raw = row.get(span_col)
            if not text:
                continue

            spans = parse_span_field(spans_raw)
            tokens, labels = span_to_bio(text, spans)
            labels = [normalise_label(l) for l in labels]

            if tokens:
                records.append({"tokens": tokens, "labels": labels, "source": source})
    return records


# ---------------------------------------------------------------------------
# finer-139 label names (279 labels = 139 entity types in BIO)
# We load these dynamically from the first file's feature metadata if available,
# otherwise fall through to LABEL_NORM which maps unknowns to FINANCIAL_ENTITY.
# ---------------------------------------------------------------------------

FINER_LABEL_NAMES = None  # populated lazily from first row if labels are ints

def read_finer_jsonl(filepath: Path) -> list:
    """
    Special reader for finer-139 where ner_tags are integer ClassLabels
    but we don't have the label list at hand. We detect and handle it.
    """
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tokens = row.get("tokens")
            ner_tags = row.get("ner_tags")
            if not tokens or ner_tags is None:
                continue

            # finer-139 stores tags as ints; we don't have the name list in JSONL.
            # Tag 0 = O, odd = B-*, even(>0) = I-*  (IOB2, 279 labels for 139 types)
            # We can't decode the specific XBRL name without the feature metadata,
            # so we map all non-O tags to FINANCIAL_ENTITY.
            labels = []
            prev_nonzero = False
            for tag in ner_tags:
                if tag == 0:
                    labels.append("O")
                    prev_nonzero = False
                elif tag % 2 == 1:  # B- tags are odd indices (1,3,5,...)
                    labels.append("B-FINANCIAL_ENTITY")
                    prev_nonzero = True
                else:  # I- tags are even indices (2,4,6,...)
                    labels.append("I-FINANCIAL_ENTITY")
                    prev_nonzero = True

            records.append({"tokens": tokens, "labels": labels, "source": "finer_139"})
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_unique_entities(all_records: list) -> dict:
    """
    Returns {source: set_of_canonical_entity_types} and a global set.
    """
    per_source = defaultdict(set)
    global_types = set()

    for rec in all_records:
        source = rec["source"]
        for label in rec["labels"]:
            if label != "O" and label.startswith("B-"):
                entity_type = label[2:]
                per_source[source].add(entity_type)
                global_types.add(entity_type)

    return per_source, global_types


def main(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    load_errors = []

    # ------------------------------------------------------------------
    # 1. ai4privacy/pii-masking-400k
    #    tokens: mbert_tokens  |  labels: mbert_token_classes (BIO strings)
    # ------------------------------------------------------------------
    p = data_dir / "ai4privacy_400k" / "train.jsonl"
    if p.exists():
        print(f"Reading {p} ...")
        recs = read_bio_jsonl(p, "mbert_tokens", "mbert_token_classes", "ai4privacy_400k")
        print(f"  {len(recs):,} records")
        all_records.extend(recs)
    else:
        load_errors.append(str(p))

    # ------------------------------------------------------------------
    # 2. ai4privacy/pii-masking-300k
    #    tokens: mbert_text_tokens  |  labels: mbert_bio_labels (BIO strings)
    # ------------------------------------------------------------------
    p = data_dir / "ai4privacy_300k" / "train.jsonl"
    if p.exists():
        print(f"Reading {p} ...")
        recs = read_bio_jsonl(p, "mbert_text_tokens", "mbert_bio_labels", "ai4privacy_300k")
        print(f"  {len(recs):,} records")
        all_records.extend(recs)
    else:
        load_errors.append(str(p))

    # ------------------------------------------------------------------
    # 3. gretelai/synthetic_pii_finance_multilingual
    #    text: generated_text  |  spans: pii_spans (JSON char offsets)
    # ------------------------------------------------------------------
    for split in ["train", "test"]:
        p = data_dir / "gretel_finance" / f"{split}.jsonl"
        if p.exists():
            print(f"Reading {p} ...")
            recs = read_span_jsonl(p, "generated_text", "pii_spans", "gretel_finance")
            print(f"  {len(recs):,} records")
            all_records.extend(recs)

    # ------------------------------------------------------------------
    # 4. nvidia/Nemotron-PII
    #    text: text  |  spans: spans (JSON char offsets)
    # ------------------------------------------------------------------
    p = data_dir / "nvidia_nemotron" / "train.jsonl"
    if p.exists():
        print(f"Reading {p} ...")
        recs = read_span_jsonl(p, "text", "spans", "nvidia_nemotron")
        print(f"  {len(recs):,} records")
        all_records.extend(recs)
    else:
        load_errors.append(str(p))

    # ------------------------------------------------------------------
    # 5. wikiann (en)
    #    tokens: tokens  |  labels: ner_tags (ClassLabel integers)
    #    label names: ['O','B-PER','I-PER','B-ORG','I-ORG','B-LOC','I-LOC']
    # ------------------------------------------------------------------
    wikiann_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    for split in ["train", "validation", "test"]:
        p = data_dir / "wikiann" / f"{split}.jsonl"
        if p.exists():
            print(f"Reading {p} ...")
            recs = read_bio_jsonl(p, "tokens", "ner_tags", "wikiann", label_names=wikiann_labels)
            print(f"  {len(recs):,} records")
            all_records.extend(recs)

    # ------------------------------------------------------------------
    # 6. Babelscape/multinerd (en only, already filtered)
    #    tokens: tokens  |  labels: ner_tags (raw integers)
    # ------------------------------------------------------------------
    multinerd_labels = [
        "O",
        "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
        "B-ANIM", "I-ANIM", "B-BIO", "I-BIO", "B-CEL", "I-CEL",
        "B-DIS", "I-DIS", "B-EVE", "I-EVE", "B-FOOD", "I-FOOD",
        "B-INST", "I-INST", "B-MEDIA", "I-MEDIA", "B-MYTH", "I-MYTH",
        "B-PLANT", "I-PLANT", "B-TIME", "I-TIME", "B-VEHI", "I-VEHI",
    ]
    p = data_dir / "multinerd" / "train_en.jsonl"
    if p.exists():
        print(f"Reading {p} ...")
        recs = read_bio_jsonl(p, "tokens", "ner_tags", "multinerd", label_names=multinerd_labels)
        print(f"  {len(recs):,} records")
        all_records.extend(recs)
    else:
        load_errors.append(str(p))

    # ------------------------------------------------------------------
    # 7. DFKI-SLT/few-nerd (supervised)
    #    tokens: tokens  |  labels: ner_tags (ClassLabel integers)
    #    coarse labels only (8 types); fine_ner_tags skipped
    # ------------------------------------------------------------------
    fewnerd_labels = ["O", "art", "building", "event", "location",
                      "organization", "other", "person", "product"]
    for split in ["train", "validation", "test"]:
        p = data_dir / "few_nerd" / f"{split}.jsonl"
        if p.exists():
            print(f"Reading {p} ...")
            recs = read_bio_jsonl(p, "tokens", "ner_tags", "few_nerd", label_names=fewnerd_labels)
            print(f"  {len(recs):,} records")
            all_records.extend(recs)

    # ------------------------------------------------------------------
    # 8. CoNLL-2003
    #    tokens: tokens  |  labels: ner_tags (ClassLabel integers)
    # ------------------------------------------------------------------
    conll_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG",
                    "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    for split in ["train", "validation", "test"]:
        p = data_dir / "conll2003" / f"{split}.jsonl"
        if p.exists():
            print(f"Reading {p} ...")
            recs = read_bio_jsonl(p, "tokens", "ner_tags", "conll2003", label_names=conll_labels)
            print(f"  {len(recs):,} records")
            all_records.extend(recs)

    # ------------------------------------------------------------------
    # 9. nlpaueb/finer-139
    #    tokens: tokens  |  labels: ner_tags (ClassLabel integers, 279 labels)
    #    All non-O mapped to FINANCIAL_ENTITY (XBRL tags, not readable as strings)
    # ------------------------------------------------------------------
    for split in ["train", "validation", "test"]:
        p = data_dir / "finer_139" / f"{split}.jsonl"
        if p.exists():
            print(f"Reading {p} ...")
            recs = read_finer_jsonl(p)
            print(f"  {len(recs):,} records")
            all_records.extend(recs)

    # ------------------------------------------------------------------
    # 10. Isotonic/pii-masking-200k
    #     tokens: tokenised_text  |  labels: bio_labels (BIO strings)
    # ------------------------------------------------------------------
    p = data_dir / "isotonic_pii_200k" / "train.jsonl"
    if p.exists():
        print(f"Reading {p} ...")
        recs = read_bio_jsonl(p, "tokenised_text", "bio_labels", "isotonic_pii_200k")
        print(f"  {len(recs):,} records")
        all_records.extend(recs)
    else:
        load_errors.append(str(p))

    # ------------------------------------------------------------------
    # Report: unique entities
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("UNIQUE ENTITY TYPES PER SOURCE")
    print("=" * 70)
    per_source, global_types = collect_unique_entities(all_records)

    for source, types in sorted(per_source.items()):
        print(f"\n[{source}] ({len(types)} types)")
        print("  " + ", ".join(sorted(types)))

    print("\n" + "=" * 70)
    print(f"GLOBAL UNIQUE ENTITY TYPES ({len(global_types)} total)")
    print("=" * 70)
    for t in sorted(global_types):
        print(f"  {t}")

    # Save entity report
    entity_report = {
        "global": sorted(global_types),
        "per_source": {src: sorted(types) for src, types in per_source.items()},
    }
    report_path = output_dir / "entity_types.json"
    with open(report_path, "w") as f:
        json.dump(entity_report, f, indent=2)
    print(f"\nEntity report saved to: {report_path}")

    # ------------------------------------------------------------------
    # Write consolidated JSONL
    # ------------------------------------------------------------------
    out_path = output_dir / "consolidated.jsonl"
    print(f"\nWriting {len(all_records):,} records to {out_path} ...")
    with open(out_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Done. File size: {size_mb:.1f} MB")

    if load_errors:
        print(f"\nWARNING — {len(load_errors)} file(s) not found (skipped):")
        for e in load_errors:
            print(f"  {e}")

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONSOLIDATION SUMMARY")
    print("=" * 70)
    source_counts = defaultdict(int)
    for rec in all_records:
        source_counts[rec["source"]] += 1
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:<40} {count:>10,} records")
    print(f"  {'TOTAL':<40} {len(all_records):>10,} records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./pii_datasets",
                        help="Root directory containing downloaded dataset folders")
    parser.add_argument("--output-dir", type=str, default="./pii_datasets/consolidated",
                        help="Directory to write consolidated output")
    args = parser.parse_args()

    main(Path(args.data_dir), Path(args.output_dir))

