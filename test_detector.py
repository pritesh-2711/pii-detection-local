"""
Test suite for PII detection model.

Tests cover:
    - PIIResult shape and field types
    - Single-text detection and redaction
    - Batch detection (PIIDetector and FastPIIDetector)
    - Edge cases: empty input, non-string input, oversized input
    - Exception types from exceptions.py
    - Statistics aggregation
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from inference import PIIDetector, FastPIIDetector, PIIResult
from exceptions import (
    EmptyInputError,
    InputTooLargeError,
    InvalidInputTypeError,
    ModelNotFoundError,
)

MODEL_PATH = "./models/best_model"

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = ""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    if condition:
        passed += 1
    else:
        failed += 1


def section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Test groups
# ---------------------------------------------------------------------------

def test_result_shape(detector: PIIDetector):
    section("PIIResult — field types and shape")

    result = detector.detect("My name is Alice and my email is alice@example.com")

    check("returns PIIResult instance",      isinstance(result, PIIResult))
    check("has_pii is bool",                 isinstance(result.has_pii, bool))
    check("redacted_text is str",            isinstance(result.redacted_text, str))
    check("pii_types is list",               isinstance(result.pii_types, list))
    check("entities is list",                isinstance(result.entities, list))
    check("error is None on success",        result.error is None)
    check("to_dict() returns dict",          isinstance(result.to_dict(), dict))

    if result.entities:
        e = result.entities[0]
        check("entity has 'text' key",       "text"       in e)
        check("entity has 'type' key",       "type"       in e)
        check("entity has 'start' key",      "start"      in e)
        check("entity has 'end' key",        "end"        in e)
        check("entity has 'confidence' key", "confidence" in e)
        check("start < end",                 e["start"] < e["end"])
        check("confidence in [0, 1]",        0.0 <= e["confidence"] <= 1.0)


def test_detection(detector: PIIDetector):
    section("Single-text detection")

    cases = [
        {
            "name":         "email",
            "text":         "Contact me at john.doe@example.com",
            "expect_pii":   True,
            "expect_types": {"EMAIL"},
        },
        {
            "name":         "person name",
            "text":         "My name is Sarah Johnson",
            "expect_pii":   True,
            "expect_types": {"PERSON"},
        },
        {
            "name":         "phone number",
            "text":         "Call me at 555-123-4567",
            "expect_pii":   True,
            "expect_types": {"PHONE"},
        },
        {
            "name":         "SSN",
            "text":         "My SSN is 123-45-6789",
            "expect_pii":   True,
            "expect_types": {"SSN"},
        },
        {
            "name":         "multiple types",
            "text":         "John Smith, email: jsmith@company.com, phone: 555-0100",
            "expect_pii":   True,
            "expect_types": {"PERSON", "EMAIL", "PHONE"},
        },
        {
            "name":         "organisation",
            "text":         "I work at Google and Microsoft",
            "expect_pii":   True,
            "expect_types": {"ORG"},
        },
        {
            "name":         "no PII",
            "text":         "The weather is nice today and I enjoy programming",
            "expect_pii":   False,
            "expect_types": set(),
        },
    ]

    for case in cases:
        result = detector.detect(case["text"])
        detected_types = set(result.pii_types)
        overlap = detected_types & case["expect_types"]

        check(
            f"{case['name']} — has_pii={case['expect_pii']}",
            result.has_pii == case["expect_pii"],
            f"got has_pii={result.has_pii}",
        )
        if case["expect_pii"]:
            check(
                f"{case['name']} — expected types present",
                bool(overlap),
                f"expected any of {case['expect_types']}, got {detected_types}",
            )


def test_redaction(detector: PIIDetector):
    section("Redaction")

    text   = "Contact Alice at alice@example.com or call 555-0199"
    result = detector.detect(text)

    if result.has_pii:
        check("redacted_text differs from original",
              result.redacted_text != text)
        check("redacted_text contains [REDACTED]",
              "[REDACTED]" in result.redacted_text)
        for entity in result.entities:
            check(
                f"entity '{entity['text']}' absent from redacted output",
                entity["text"] not in result.redacted_text,
            )
    else:
        check("no PII — redacted_text equals original",
              result.redacted_text == text)

    clean = "The quarterly revenue increased by 12 percent."
    check("clean text — redacted_text equals original",
          detector.detect(clean).redacted_text == clean)


def test_edge_cases(detector: PIIDetector):
    section("Edge cases and exception handling")

    # Empty string
    try:
        detector.detect("")
        check("empty string raises EmptyInputError", False, "no exception raised")
    except EmptyInputError:
        check("empty string raises EmptyInputError", True)
    except Exception as exc:
        check("empty string raises EmptyInputError", False, f"got {type(exc).__name__}")

    # Whitespace-only string
    try:
        detector.detect("   \t\n  ")
        check("whitespace-only raises EmptyInputError", False, "no exception raised")
    except EmptyInputError:
        check("whitespace-only raises EmptyInputError", True)
    except Exception as exc:
        check("whitespace-only raises EmptyInputError", False, f"got {type(exc).__name__}")

    # Non-string input
    try:
        detector.detect(12345)          # type: ignore
        check("non-string raises InvalidInputTypeError", False, "no exception raised")
    except InvalidInputTypeError:
        check("non-string raises InvalidInputTypeError", True)
    except Exception as exc:
        check("non-string raises InvalidInputTypeError", False, f"got {type(exc).__name__}")

    # Oversized input
    try:
        detector.detect("a" * 60_000)
        check("oversized input raises InputTooLargeError", False, "no exception raised")
    except InputTooLargeError:
        check("oversized input raises InputTooLargeError", True)
    except Exception as exc:
        check("oversized input raises InputTooLargeError", False, f"got {type(exc).__name__}")

    # Bad model path
    try:
        PIIDetector(model_path="./models/does_not_exist")
        check("bad model path raises ModelNotFoundError", False, "no exception raised")
    except ModelNotFoundError:
        check("bad model path raises ModelNotFoundError", True)
    except Exception as exc:
        check("bad model path raises ModelNotFoundError", False, f"got {type(exc).__name__}")


def test_batch_sequential(detector: PIIDetector):
    section("Batch detection — PIIDetector (sequential)")

    # Empty list must raise
    try:
        detector.batch_detect([])
        check("empty list raises EmptyInputError", False, "no exception raised")
    except EmptyInputError:
        check("empty list raises EmptyInputError", True)
    except Exception as exc:
        check("empty list raises EmptyInputError", False, f"got {type(exc).__name__}")

    texts = [
        "My name is Alice",
        "The sky is blue",
        "Call Bob at 555-0100",
        "",                         # captured as error, not raised
        42,                         # type: ignore — same
        "SSN: 123-45-6789",
    ]
    results = detector.batch_detect(texts)

    check("result length matches input",       len(results) == len(texts))
    check("all results are PIIResult",         all(isinstance(r, PIIResult) for r in results))
    check("empty-string result has error set", results[3].error is not None)
    check("non-string result has error set",   results[4].error is not None)
    check("valid texts have no error",         results[0].error is None)


def test_batch_fast(detector: FastPIIDetector):
    section("Batch detection — FastPIIDetector (padded batches)")

    texts = [
        "My name is Alice Johnson and my email is alice@example.com",
        "The quarterly revenue increased by 12 percent.",
        "Wire transfer from account 50100123456789",
        "Call me at +91-98765-43210",
        "No personal information here at all.",
    ]
    results = detector.batch_detect(texts)

    check("result length matches input",  len(results) == len(texts))
    check("all PIIResult instances",      all(isinstance(r, PIIResult) for r in results))
    check("no unexpected errors",         all(r.error is None for r in results))
    check("all have redacted_text str",   all(isinstance(r.redacted_text, str) for r in results))

    pii_results = [r for r, t in zip(results, texts)
                   if "alice@example.com" in t or "98765" in t]
    check("PII texts flagged as has_pii", any(r.has_pii for r in pii_results))


def test_statistics(detector: PIIDetector):
    section("Statistics aggregation")

    texts = [
        "My name is Alice and my email is alice@example.com",
        "The sky is blue today.",
        "SSN: 123-45-6789",
        "No PII in this sentence at all.",
        "Call 555-0100 or email bob@corp.com",
    ]
    results = detector.batch_detect(texts)
    stats   = detector.get_pii_statistics(results)

    check("has 'total_texts'",           "total_texts"           in stats)
    check("has 'texts_with_pii'",        "texts_with_pii"        in stats)
    check("has 'texts_without_pii'",     "texts_without_pii"     in stats)
    check("has 'pii_rate'",              "pii_rate"              in stats)
    check("has 'pii_type_distribution'", "pii_type_distribution" in stats)
    check("has 'errors'",                "errors"                in stats)
    check("total_texts == len(texts)",   stats["total_texts"] == len(texts))
    check("with + without == total",
          stats["texts_with_pii"] + stats["texts_without_pii"] == stats["total_texts"])
    check("pii_rate in [0, 1]",          0.0 <= stats["pii_rate"] <= 1.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PII DETECTOR — TEST SUITE")
    print("=" * 70)

    print("\nLoading PIIDetector ...")
    try:
        detector = PIIDetector(model_path=MODEL_PATH)
    except Exception as exc:
        print(f"FATAL: Could not load PIIDetector: {exc}")
        print("Run the training pipeline first: python run_training_pipeline.py")
        sys.exit(1)

    print("Loading FastPIIDetector ...")
    try:
        fast_detector = FastPIIDetector(model_path=MODEL_PATH, batch_size=4)
    except Exception as exc:
        print(f"FATAL: Could not load FastPIIDetector: {exc}")
        sys.exit(1)

    test_result_shape(detector)
    test_detection(detector)
    test_redaction(detector)
    test_edge_cases(detector)
    test_batch_sequential(detector)
    test_batch_fast(fast_detector)
    test_statistics(detector)

    total = passed + failed
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {passed}/{total} passed  |  {failed} failed")
    print("=" * 70)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()