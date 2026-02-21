#!/usr/bin/env python3
"""
Example client for the PII Detection API.

Demonstrates all four endpoints:
    GET  /health
    GET  /info
    POST /detect          single text
    POST /detect/batch    list of texts with optional statistics
    POST /detect/file     file upload (.txt, .csv, .xlsx, .xls)

Usage:
    # Start the server first:
    python src/api.py --model-path ./models/best_model

    # Run the client:
    python example_client.py
    python example_client.py --host http://localhost:5000
"""

import sys
import json
import argparse
import tempfile
import os
from pathlib import Path

import requests


class PIIDetectionClient:
    """HTTP client for the PII Detection API."""

    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip("/")
        self.session  = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health_check(self) -> dict:
        resp = self.session.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_info(self) -> dict:
        resp = self.session.get(f"{self.base_url}/info", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def detect(self, text: str) -> dict:
        """
        Single text detection.

        Always returns a PIIResult dict:
            has_pii, redacted_text, pii_types, entities, error
        """
        resp = self.session.post(
            f"{self.base_url}/detect",
            json={"text": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def detect_batch(self, texts: list, return_stats: bool = True) -> dict:
        """
        Batch detection.

        Returns {"results": [...]} or {"results": [...], "statistics": {...}}
        when return_stats=True. Every entry in results is a PIIResult dict —
        never None, even for texts with no PII.
        """
        resp = self.session.post(
            f"{self.base_url}/detect/batch",
            json={"texts": texts, "return_stats": return_stats},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def detect_file(self, file_path: str, columns: list = None) -> dict:
        """
        File upload detection for .txt, .csv, .xlsx, .xls files.

        columns: optional list of column names to process (CSV/Excel only).
                 Omit to process all columns.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        form_data = {}
        if columns:
            form_data["columns"] = ",".join(columns)

        with open(file_path, "rb") as f:
            resp = requests.post(
                f"{self.base_url}/detect/file",
                files={"file": (file_path.name, f)},
                data=form_data,
                timeout=120,
            )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_result(result: dict, indent: int = 4):
    pad = " " * indent
    print(f"{pad}has_pii      : {result['has_pii']}")
    print(f"{pad}redacted_text: {result['redacted_text']}")
    if result.get("pii_types"):
        print(f"{pad}pii_types    : {', '.join(result['pii_types'])}")
    if result.get("entities"):
        print(f"{pad}entities     :")
        for e in result["entities"]:
            print(
                f"{pad}  [{e['start']}:{e['end']}] "
                f"{e['text']!r:<28} | {e['type']:<20} | conf={e['confidence']:.2f}"
            )
    if result.get("error"):
        print(f"{pad}error        : {result['error']}")


def print_stats(stats: dict, indent: int = 4):
    pad = " " * indent
    print(f"{pad}total_texts      : {stats['total_texts']}")
    print(f"{pad}texts_with_pii   : {stats['texts_with_pii']}")
    print(f"{pad}texts_without_pii: {stats['texts_without_pii']}")
    print(f"{pad}pii_rate         : {stats['pii_rate']:.1%}")
    if stats.get("pii_type_distribution"):
        print(f"{pad}type_distribution: {stats['pii_type_distribution']}")


# ---------------------------------------------------------------------------
# Demo sections
# ---------------------------------------------------------------------------

def demo_health(client: PIIDetectionClient):
    print("\n[1] Health Check")
    print("-" * 50)
    health = client.health_check()
    print(f"    status      : {health['status']}")
    print(f"    model_loaded: {health['model_loaded']}")


def demo_info(client: PIIDetectionClient):
    print("\n[2] Model Info")
    print("-" * 50)
    info = client.get_info()
    print(f"    model_path          : {info['model_path']}")
    print(f"    device              : {info['device']}")
    print(f"    confidence_threshold: {info['confidence_threshold']}")
    print(f"    batch_size          : {info['batch_size']}")
    print(f"    num_labels          : {info['num_labels']}")
    types = info.get("supported_pii_types", [])
    print(f"    supported_pii_types : {', '.join(types[:6])}{'...' if len(types) > 6 else ''}")


def demo_detect(client: PIIDetectionClient):
    print("\n[3] Single Text Detection")
    print("-" * 50)
    text = "My name is Alice Johnson and my email is alice@example.com"
    print(f"    text: {text}")
    result = client.detect(text)
    print_result(result)


def demo_batch(client: PIIDetectionClient):
    print("\n[4] Batch Detection")
    print("-" * 50)
    texts = [
        "Contact support at help@company.com",
        "The weather is sunny today",
        "My SSN is 123-45-6789",
        "Call Bob at 555-0100",
        "The quarterly revenue increased by 12 percent.",
    ]
    print(f"    submitting {len(texts)} texts ...")
    response = client.detect_batch(texts, return_stats=True)

    for i, result in enumerate(response["results"]):
        status = "PII" if result["has_pii"] else "clean"
        types  = ", ".join(result["pii_types"]) if result["pii_types"] else "-"
        print(f"    [{i+1}] {status:<6}  types={types}")
        if result.get("error"):
            print(f"         error: {result['error']}")

    if "statistics" in response:
        print("\n    Statistics:")
        print_stats(response["statistics"], indent=8)


def demo_file_txt(client: PIIDetectionClient):
    print("\n[5] File Upload — plain text")
    print("-" * 50)

    content = "\n".join([
        "Alice Johnson lives at 42 Elm Street, Boston MA.",
        "Her email is alice@example.com and SSN is 123-45-6789.",
        "The quarterly earnings report shows a 12% increase.",
        "Contact Bob Smith at bob.smith@corp.com for details.",
    ])

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        response = client.detect_file(tmp_path)
        print(f"    filename : {response['filename']}")
        print(f"    file_type: {response['file_type']}")
        for row in response["results"]:
            status = "PII" if row["has_pii"] else "clean"
            print(f"    line {row['line']:>2}: {status:<6}  {row['redacted_text'][:60]}")
        print("\n    Statistics:")
        print_stats(response["statistics"], indent=8)
    finally:
        os.unlink(tmp_path)


def demo_file_csv(client: PIIDetectionClient):
    print("\n[6] File Upload — CSV (selected columns)")
    print("-" * 50)

    csv_content = (
        "name,department,email,notes\n"
        "Alice Johnson,Engineering,alice@example.com,Team lead\n"
        "Bob Smith,Finance,bob.smith@corp.com,Budget review\n"
        "Carol White,HR,carol@hr.org,No issues\n"
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(csv_content)
        tmp_path = tmp.name

    try:
        # Process only the 'name' and 'email' columns
        response = client.detect_file(tmp_path, columns=["name", "email"])
        print(f"    filename         : {response['filename']}")
        print(f"    columns_processed: {response['columns_processed']}")
        print(f"    total cell results: {len(response['results'])}")
        for row in response["results"]:
            status = "PII" if row["has_pii"] else "clean"
            print(
                f"    row={row['row']} col={row['column']:<8} "
                f"{status:<6}  redacted='{row['redacted_value']}'"
            )
        print("\n    First redacted record:")
        if response["redacted_records"]:
            print(f"    {json.dumps(response['redacted_records'][0], indent=6)}")
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PII Detection API example client")
    parser.add_argument(
        "--host",
        default="http://localhost:5000",
        help="API base URL (default: http://localhost:5000)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PII DETECTION API — EXAMPLE CLIENT")
    print("=" * 70)
    print(f"Connecting to: {args.host}")

    client = PIIDetectionClient(base_url=args.host)

    try:
        demo_health(client)
        demo_info(client)
        demo_detect(client)
        demo_batch(client)
        demo_file_txt(client)
        demo_file_csv(client)
    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Could not connect to {args.host}")
        print("Make sure the server is running: python src/api.py --model-path ./models/best_model")
        sys.exit(1)
    except requests.exceptions.HTTPError as exc:
        print(f"\nHTTP ERROR {exc.response.status_code}: {exc.response.text}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()