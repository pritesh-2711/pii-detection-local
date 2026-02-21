"""
PII Detection REST API.

Endpoints:
    GET  /health                  — liveness + model status
    GET  /info                    — model metadata
    POST /detect                  — single text
    POST /detect/batch            — list of strings
    POST /detect/file             — .txt / .csv / .xlsx / .xls upload
                                    CSV/Excel: optional `columns` param to
                                    restrict which columns are processed.

All error responses follow the shape:
    {"error": "<ExceptionClassName>", "message": "<human-readable>", "details": {...}}
"""

import io
import os
import chardet
import argparse
from pathlib import Path

import pandas as pd
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

import sys
sys.path.append(str(Path(__file__).parent))

from inference import PIIDetector, FastPIIDetector, PIIResult
from exceptions import (
    PIIDetectionError,
    ModelNotFoundError,
    ModelLoadError,
    EmptyInputError,
    InputTooLargeError,
    InvalidInputTypeError,
    UnsupportedFileTypeError,
    FileReadError,
    TextFileParsingError,
    CSVParsingError,
    ExcelParsingError,
    MissingFieldError,
    InvalidFieldError,
)

app = Flask(__name__)
CORS(app)

detector: FastPIIDetector = None

# Maximum number of texts accepted in a single batch request.
MAX_BATCH_SIZE = 1_000

# Maximum file size in bytes (50 MB).
MAX_FILE_BYTES = 50 * 1024 * 1024

# Supported file extensions and their MIME types.
SUPPORTED_EXTENSIONS = {".txt", ".csv", ".xlsx", ".xls"}


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialize_detector(model_path: str, batch_size: int = 32):
    global detector
    try:
        detector = FastPIIDetector(
            model_path=model_path,
            confidence_threshold=0.5,
            batch_size=batch_size,
        )
        print(f"Detector initialised from {model_path}")
    except (ModelNotFoundError, ModelLoadError):
        raise
    except Exception as exc:
        raise ModelLoadError(model_path, str(exc)) from exc


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def _error_response(exc: PIIDetectionError, status: int) -> Response:
    return jsonify(exc.to_dict()), status


def _generic_error(message: str, status: int = 500) -> Response:
    return jsonify({"error": "InternalServerError", "message": message}), status


# ---------------------------------------------------------------------------
# File parsers
# ---------------------------------------------------------------------------

def _detect_encoding(raw: bytes) -> str:
    detected = chardet.detect(raw)
    return detected.get("encoding") or "utf-8"


def _parse_txt(filename: str, raw: bytes) -> list[str]:
    """
    Parse a plain-text file into non-blank lines.
    Each line becomes one text unit for inference.
    """
    try:
        encoding = _detect_encoding(raw)
        text = raw.decode(encoding, errors="replace")
    except Exception as exc:
        raise TextFileParsingError(filename, f"Decoding failed: {exc}") from exc

    lines = [line.strip() for line in text.splitlines()]
    lines = [l for l in lines if l]

    if not lines:
        raise EmptyInputError(f"text file '{filename}'")

    return lines


def _parse_csv(filename: str, raw: bytes, columns: list[str] = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Parse a CSV file.

    Returns (dataframe, text_columns) where text_columns is the resolved list
    of column names that will be processed for PII.
    """
    try:
        encoding = _detect_encoding(raw)
        df = pd.read_csv(io.BytesIO(raw), encoding=encoding, dtype=str)
    except Exception as exc:
        raise CSVParsingError(filename, str(exc)) from exc

    if df.empty:
        raise EmptyInputError(f"CSV file '{filename}'")

    return _resolve_columns(df, columns, filename, "CSV")


def _parse_excel(filename: str, raw: bytes, columns: list[str] = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Parse an Excel (.xlsx / .xls) file.

    Returns (dataframe, text_columns).
    """
    try:
        df = pd.read_excel(io.BytesIO(raw), dtype=str)
    except Exception as exc:
        raise ExcelParsingError(filename, str(exc)) from exc

    if df.empty:
        raise EmptyInputError(f"Excel file '{filename}'")

    return _resolve_columns(df, columns, filename, "Excel")


def _resolve_columns(
    df: pd.DataFrame,
    columns: list[str],
    filename: str,
    filetype: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Validate and resolve the list of columns to process.
    If columns is None or empty, all columns are used.
    """
    all_cols = list(df.columns)

    if not columns:
        return df, all_cols

    missing = [c for c in columns if c not in all_cols]
    if missing:
        raise InvalidFieldError(
            "columns",
            f"Column(s) {missing} not found in {filetype} file '{filename}'. "
            f"Available columns: {all_cols}",
        )

    return df, columns


# ---------------------------------------------------------------------------
# Result serialisation helpers
# ---------------------------------------------------------------------------

def _result_to_dict(result: PIIResult) -> dict:
    return result.to_dict()


def _file_row_result(
    row_index: int,
    column: str,
    original_value: str,
    result: PIIResult,
) -> dict:
    return {
        "row":            row_index,
        "column":         column,
        "original_value": original_value,
        "has_pii":        result.has_pii,
        "redacted_value": result.redacted_text,
        "pii_types":      result.pii_types,
        "entities":       result.entities,
        "error":          result.error,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status":       "healthy",
        "model_loaded": detector is not None,
    }), 200


@app.route("/info", methods=["GET"])
def model_info():
    if detector is None:
        return _error_response(
            ModelNotFoundError("(not initialised)"), 503
        )
    return jsonify({
        "model_path":          str(detector.model_path),
        "device":              str(detector.device),
        "confidence_threshold": detector.confidence_threshold,
        "supported_pii_types": detector.pii_types,
        "num_labels":          len(detector.id2label),
        "batch_size":          detector.batch_size,
    }), 200


@app.route("/detect", methods=["POST"])
def detect_single():
    """
    Detect and redact PII in a single string.

    Request body:
        {"text": "some string"}

    Response:
        {
            "has_pii":       true | false,
            "redacted_text": "...",
            "pii_types":     [...],
            "entities":      [...],
            "error":         null
        }
    """
    if detector is None:
        return _error_response(ModelNotFoundError("(not initialised)"), 503)

    try:
        body = request.get_json(silent=True)
        if body is None:
            raise InvalidFieldError("request body", "must be valid JSON")
        if "text" not in body:
            raise MissingFieldError("text")

        text = body["text"]
        if not isinstance(text, str):
            raise InvalidFieldError("text", f"expected string, got {type(text).__name__}")
        if not text.strip():
            raise EmptyInputError("text")

        result = detector.detect(text)
        return jsonify(_result_to_dict(result)), 200

    except PIIDetectionError as exc:
        status = 400 if not isinstance(exc, PIIDetectionError.__bases__[0]) else 400
        # Map specific types to appropriate HTTP codes
        if isinstance(exc, (MissingFieldError, InvalidFieldError,
                             EmptyInputError, InputTooLargeError,
                             InvalidInputTypeError)):
            status = 400
        elif isinstance(exc, (ModelNotFoundError, ModelLoadError)):
            status = 503
        else:
            status = 400
        return _error_response(exc, status)

    except Exception as exc:
        return _generic_error(str(exc), 500)


@app.route("/detect/batch", methods=["POST"])
def detect_batch():
    """
    Detect and redact PII in a list of strings.

    Request body:
        {"texts": ["string1", "string2", ...]}

    Optional:
        {"texts": [...], "return_stats": true}

    Response:
        {
            "results":    [{...}, ...],   one PIIResult per input text
            "statistics": {...}           present when return_stats is true
        }
    """
    if detector is None:
        return _error_response(ModelNotFoundError("(not initialised)"), 503)

    try:
        body = request.get_json(silent=True)
        if body is None:
            raise InvalidFieldError("request body", "must be valid JSON")
        if "texts" not in body:
            raise MissingFieldError("texts")

        texts = body["texts"]
        if not isinstance(texts, list):
            raise InvalidFieldError("texts", f"expected list, got {type(texts).__name__}")
        if not texts:
            raise EmptyInputError("texts list")
        if len(texts) > MAX_BATCH_SIZE:
            raise InvalidFieldError(
                "texts",
                f"list length {len(texts)} exceeds maximum allowed {MAX_BATCH_SIZE}",
            )

        return_stats = bool(body.get("return_stats", False))

        results = detector.batch_detect(texts)
        response = {"results": [_result_to_dict(r) for r in results]}

        if return_stats:
            response["statistics"] = detector.get_pii_statistics(results)

        return jsonify(response), 200

    except PIIDetectionError as exc:
        status = 400 if isinstance(exc, (MissingFieldError, InvalidFieldError,
                                         EmptyInputError)) else 503
        return _error_response(exc, status)

    except Exception as exc:
        return _generic_error(str(exc), 500)


@app.route("/detect/file", methods=["POST"])
def detect_file():
    """
    Detect and redact PII from an uploaded file.

    Supported file types: .txt, .csv, .xlsx, .xls

    Multipart form fields:
        file     (required)  — the uploaded file
        columns  (optional)  — comma-separated column names to process
                               (CSV/Excel only; defaults to all columns)

    Response for .txt:
        {
            "filename":   "...",
            "file_type":  "txt",
            "results":    [
                {
                    "line":          1,
                    "original_text": "...",
                    "has_pii":       true | false,
                    "redacted_text": "...",
                    "pii_types":     [...],
                    "entities":      [...],
                    "error":         null
                },
                ...
            ],
            "statistics": {...}
        }

    Response for .csv / .xlsx / .xls:
        {
            "filename":         "...",
            "file_type":        "csv" | "excel",
            "columns_processed": [...],
            "results":          [
                {
                    "row":            0,
                    "column":         "col_name",
                    "original_value": "...",
                    "has_pii":        true | false,
                    "redacted_value": "...",
                    "pii_types":      [...],
                    "entities":       [...],
                    "error":          null
                },
                ...
            ],
            "redacted_records": [{"col1": "...", "col2": "..."}, ...],
            "statistics":       {...}
        }
    """
    if detector is None:
        return _error_response(ModelNotFoundError("(not initialised)"), 503)

    try:
        if "file" not in request.files:
            raise MissingFieldError("file")

        upload = request.files["file"]
        filename = upload.filename or "upload"

        raw = upload.read()
        if not raw:
            raise EmptyInputError(f"uploaded file '{filename}'")
        if len(raw) > MAX_FILE_BYTES:
            raise InvalidFieldError(
                "file",
                f"file size {len(raw) / 1024 / 1024:.1f} MB exceeds maximum "
                f"{MAX_FILE_BYTES // 1024 // 1024} MB",
            )

        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(filename, ext)

        # Parse columns parameter (CSV/Excel only)
        columns_raw = request.form.get("columns", "").strip()
        columns = [c.strip() for c in columns_raw.split(",") if c.strip()] or None

        # ------------------------------------------------------------------
        # Plain text
        # ------------------------------------------------------------------
        if ext == ".txt":
            lines = _parse_txt(filename, raw)
            pii_results = detector.batch_detect(lines)

            row_results = []
            for line_num, (line, result) in enumerate(zip(lines, pii_results), start=1):
                row_results.append({
                    "line":          line_num,
                    "original_text": line,
                    "has_pii":       result.has_pii,
                    "redacted_text": result.redacted_text,
                    "pii_types":     result.pii_types,
                    "entities":      result.entities,
                    "error":         result.error,
                })

            return jsonify({
                "filename":   filename,
                "file_type":  "txt",
                "results":    row_results,
                "statistics": detector.get_pii_statistics(pii_results),
            }), 200

        # ------------------------------------------------------------------
        # CSV
        # ------------------------------------------------------------------
        if ext == ".csv":
            df, text_cols = _parse_csv(filename, raw, columns)
            return jsonify(
                _build_tabular_response(filename, "csv", df, text_cols)
            ), 200

        # ------------------------------------------------------------------
        # Excel
        # ------------------------------------------------------------------
        if ext in {".xlsx", ".xls"}:
            df, text_cols = _parse_excel(filename, raw, columns)
            return jsonify(
                _build_tabular_response(filename, "excel", df, text_cols)
            ), 200

    except PIIDetectionError as exc:
        if isinstance(exc, (MissingFieldError, InvalidFieldError,
                             EmptyInputError, UnsupportedFileTypeError,
                             FileReadError, TextFileParsingError,
                             CSVParsingError, ExcelParsingError)):
            status = 400
        elif isinstance(exc, (ModelNotFoundError, ModelLoadError)):
            status = 503
        else:
            status = 400
        return _error_response(exc, status)

    except Exception as exc:
        return _generic_error(str(exc), 500)


# ---------------------------------------------------------------------------
# Tabular response builder (shared by CSV + Excel)
# ---------------------------------------------------------------------------

def _build_tabular_response(
    filename: str,
    file_type: str,
    df: pd.DataFrame,
    text_cols: list[str],
) -> dict:
    """
    Run inference on every cell in text_cols, build per-cell results,
    and construct a redacted copy of the dataframe as redacted_records.
    """
    # Collect all (row_idx, col, value) triples to run in one batch
    triples = []
    for col in text_cols:
        for row_idx, value in enumerate(df[col].fillna("").astype(str)):
            triples.append((row_idx, col, value))

    texts    = [t[2] for t in triples]
    pii_results = detector.batch_detect(texts)

    # Per-cell result list
    cell_results = []
    for (row_idx, col, original), result in zip(triples, pii_results):
        cell_results.append(_file_row_result(row_idx, col, original, result))

    # Build redacted dataframe
    redacted_df = df.copy()
    for (row_idx, col, _), result in zip(triples, pii_results):
        redacted_df.at[row_idx, col] = result.redacted_text

    return {
        "filename":          filename,
        "file_type":         file_type,
        "columns_processed": text_cols,
        "results":           cell_results,
        "redacted_records":  redacted_df.to_dict(orient="records"),
        "statistics":        detector.get_pii_statistics(pii_results),
    }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(model_path: str = "./models/best_model", batch_size: int = 32) -> Flask:
    initialize_detector(model_path, batch_size)
    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PII Detection API Server")
    parser.add_argument("--model-path", type=str, default="./models/best_model")
    parser.add_argument("--host",       type=str, default="0.0.0.0")
    parser.add_argument("--port",       type=int, default=5000)
    parser.add_argument("--debug",      action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    initialize_detector(args.model_path, args.batch_size)
    print(f"\nStarting PII Detection API on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)