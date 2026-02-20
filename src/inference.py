"""
PII detection inference using a fine-tuned DeBERTa-v3-base model.
Backward-compatible with src/api.py interface.

Two classes:
    PIIDetector      — single-text inference with character offsets
    FastPIIDetector  — batched inference for high-throughput pipelines
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


# ---------------------------------------------------------------------------
# PIIDetector
# ---------------------------------------------------------------------------

class PIIDetector:
    """
    Loads a fine-tuned DeBERTa-v3-base token classification model.

    detect(text) returns:
        {
            "has_pii"   : True,
            "pii_types" : ["EMAIL", "PERSON", ...],
            "entities"  : [
                {
                    "text"      : "John Smith",
                    "type"      : "PERSON",
                    "start"     : 11,       # character offset in original text
                    "end"       : 21,
                    "confidence": 0.983,
                }
            ],
            "message"   : "PII detected in input text",
        }
        or None if no PII found.

    Character offsets are computed from the tokenizer's offset_mapping so they
    are exact and suitable for downstream redaction or benchmarking against
    spaCy / Presidio which also return character spans.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        self.model_path           = Path(model_path)
        self.confidence_threshold = confidence_threshold

        with open(self.model_path / "label_mapping.json") as f:
            mapping = json.load(f)
        self.label2id = mapping["label2id"]
        self.id2label = {int(k): v for k, v in mapping["id2label"].items()}

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model     = AutoModelForTokenClassification.from_pretrained(
            str(self.model_path)
        )

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

        # All entity type names (B- prefix stripped)
        self.pii_types = sorted({
            lbl[2:] for lbl in self.id2label.values()
            if lbl.startswith("B-")
        })

        print(f"PIIDetector loaded: {self.model_path} | device={self.device} | "
              f"types={len(self.pii_types)} | threshold={self.confidence_threshold}")

    @torch.inference_mode()
    def detect(self, text: str) -> Optional[Dict]:
        if not text or not text.strip():
            return None

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
            return_offsets_mapping=True,    # needed for exact character spans
        )

        # offset_mapping must be removed before passing to model
        offset_mapping = encoding.pop("offset_mapping")[0].tolist()

        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        logits      = self.model(**encoding).logits           # [1, seq, num_labels]
        probs       = torch.softmax(logits, dim=-1)[0]        # [seq, num_labels]
        pred_ids    = torch.argmax(probs, dim=-1).cpu().numpy()
        confidences = probs.max(dim=-1).values.cpu().numpy()

        entities = self._extract_entities(
            text, pred_ids, confidences, offset_mapping
        )

        if not entities:
            return None

        return {
            "has_pii":   True,
            "pii_types": sorted({e["type"] for e in entities}),
            "entities":  entities,
            "message":   "PII detected in input text",
        }

    def _extract_entities(
        self,
        text: str,
        pred_ids: np.ndarray,
        confidences: np.ndarray,
        offset_mapping: List[Tuple[int, int]],
    ) -> List[Dict]:
        """
        Reconstructs entities from BIO predictions with character-level offsets.
        Special tokens have offset (0, 0) and are skipped.
        """
        entities      = []
        current_type  = None
        current_start = None
        current_end   = None
        current_confs = []

        for pred_id, conf, (char_start, char_end) in zip(
            pred_ids, confidences, offset_mapping
        ):
            # Special tokens ([CLS], [SEP], padding) have zero-length spans
            if char_start == 0 and char_end == 0:
                continue

            label = self.id2label.get(int(pred_id), "O")

            # Apply confidence threshold — treat low-confidence predictions as O
            if float(conf) < self.confidence_threshold:
                label = "O"

            if label.startswith("B-"):
                # Flush previous entity
                if current_type is not None:
                    entities.append(self._make_entity(
                        text, current_type, current_start, current_end, current_confs
                    ))
                current_type  = label[2:]
                current_start = char_start
                current_end   = char_end
                current_confs = [float(conf)]

            elif label.startswith("I-") and current_type == label[2:]:
                # Extend current entity span
                current_end = char_end
                current_confs.append(float(conf))

            else:
                # O or I- without matching B- (broken sequence)
                if current_type is not None:
                    entities.append(self._make_entity(
                        text, current_type, current_start, current_end, current_confs
                    ))
                current_type  = None
                current_start = None
                current_end   = None
                current_confs = []

        # Flush final entity
        if current_type is not None:
            entities.append(self._make_entity(
                text, current_type, current_start, current_end, current_confs
            ))

        return entities

    def _make_entity(
        self,
        text: str,
        etype: str,
        start: int,
        end: int,
        confs: List[float],
    ) -> Dict:
        return {
            "text":       text[start:end],
            "type":       etype,
            "start":      start,
            "end":        end,
            "confidence": float(np.mean(confs)),
        }

    def batch_detect(self, texts: List[str]) -> List[Optional[Dict]]:
        """Sequential single-text inference. Use FastPIIDetector for throughput."""
        return [self.detect(t) for t in texts]

    def get_pii_statistics(self, results: List[Optional[Dict]]) -> Dict:
        total    = len(results)
        with_pii = sum(1 for r in results if r is not None)
        type_counts: Dict[str, int] = {}
        for r in results:
            if r:
                for t in r["pii_types"]:
                    type_counts[t] = type_counts.get(t, 0) + 1
        return {
            "total_texts":          total,
            "texts_with_pii":       with_pii,
            "texts_without_pii":    total - with_pii,
            "pii_rate":             round(with_pii / total, 4) if total > 0 else 0.0,
            "pii_type_distribution": dict(
                sorted(type_counts.items(), key=lambda x: -x[1])
            ),
        }

    def redact(self, text: str, replacement: str = "[REDACTED]") -> str:
        """
        Returns text with all detected PII spans replaced by `replacement`.
        Applies replacements in reverse offset order to preserve positions.
        """
        result = self.detect(text)
        if result is None:
            return text

        entities = sorted(result["entities"], key=lambda e: e["start"], reverse=True)
        out = text
        for entity in entities:
            out = out[:entity["start"]] + replacement + out[entity["end"]:]
        return out


# ---------------------------------------------------------------------------
# FastPIIDetector — batched inference
# ---------------------------------------------------------------------------

class FastPIIDetector(PIIDetector):
    """
    Batched inference for high-throughput pipelines.
    Pads a batch of texts together and runs a single forward pass per batch.
    Significantly faster than calling detect() in a loop when processing
    large volumes of text.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        super().__init__(model_path, confidence_threshold, device)
        self.batch_size = batch_size

    @torch.inference_mode()
    def batch_detect_optimized(self, texts: List[str]) -> List[Optional[Dict]]:
        if not texts:
            return []

        results = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            encoding = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
                return_offsets_mapping=True,
            )

            # Remove offset_mapping before forward pass
            offset_mappings = encoding.pop("offset_mapping").tolist()

            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            logits   = self.model(**encoding).logits        # [B, seq, num_labels]
            probs    = torch.softmax(logits, dim=-1)        # [B, seq, num_labels]

            for j, (text, offsets) in enumerate(zip(batch, offset_mappings)):
                pred_ids = torch.argmax(probs[j], dim=-1).cpu().numpy()
                confs    = probs[j].max(dim=-1).values.cpu().numpy()

                entities = self._extract_entities(
                    text, pred_ids, confs, offsets
                )

                if entities:
                    results.append({
                        "has_pii":   True,
                        "pii_types": sorted({e["type"] for e in entities}),
                        "entities":  entities,
                        "message":   "PII detected in input text",
                    })
                else:
                    results.append(None)

        return results

    def batch_redact(
        self,
        texts: List[str],
        replacement: str = "[REDACTED]",
    ) -> List[str]:
        """Redacts all PII in a list of texts using batched inference."""
        results = self.batch_detect_optimized(texts)
        redacted = []
        for text, result in zip(texts, results):
            if result is None:
                redacted.append(text)
            else:
                entities = sorted(
                    result["entities"], key=lambda e: e["start"], reverse=True
                )
                out = text
                for entity in entities:
                    out = out[:entity["start"]] + replacement + out[entity["end"]:]
                redacted.append(out)
        return redacted


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    detector = PIIDetector(model_path="./models/best_model")

    samples = [
        "My name is Pritesh Jha and my email is pritesh.jha@example.com",
        "Call me at +91-98765-43210 or reach John Doe at john.doe@corp.in",
        "The quarterly revenue increased by 12 percent.",
        "SSN: 123-45-6789, DOB: 01/15/1990, card ending 4242",
        "Wire transfer from HDFC account 50100123456789 to JP Morgan Chase",
    ]

    print("=" * 60)
    print("PII Detection — smoke test")
    print("=" * 60)

    for text in samples:
        result = detector.detect(text)
        print(f"\nText : {text}")
        if result:
            print(f"Types: {result['pii_types']}")
            for e in result["entities"]:
                print(f"  [{e['start']}:{e['end']}] {e['text']!r:30s} | "
                      f"{e['type']:25s} | conf={e['confidence']:.3f}")
        else:
            print("  No PII detected")

    print("\n" + "=" * 60)
    print("Redaction test")
    print("=" * 60)
    text = "Contact Sarah Connor at sarah@skynet.com or +1-800-555-0199"
    print(f"Original : {text}")
    print(f"Redacted : {detector.redact(text)}")