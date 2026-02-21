"""
Custom exceptions for the PII detection system.

Hierarchy:
    PIIDetectionError               — base for all project exceptions
    ├── ModelError
    │   ├── ModelNotFoundError
    │   ├── ModelLoadError
    │   └── ModelInferenceError
    ├── InputError
    │   ├── EmptyInputError
    │   ├── InputTooLargeError
    │   └── InvalidInputTypeError
    ├── FileParsingError
    │   ├── UnsupportedFileTypeError
    │   ├── FileReadError
    │   ├── TextFileParsingError
    │   ├── CSVParsingError
    │   └── ExcelParsingError
    └── APIError
        ├── MissingFieldError
        └── InvalidFieldError
"""


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class PIIDetectionError(Exception):
    """Base exception for all PII detection errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        d = {"error": self.__class__.__name__, "message": self.message}
        if self.details:
            d["details"] = self.details
        return d


# ---------------------------------------------------------------------------
# Model errors
# ---------------------------------------------------------------------------

class ModelError(PIIDetectionError):
    """Base for model-related failures."""


class ModelNotFoundError(ModelError):
    """Model directory or required files do not exist at the given path."""

    def __init__(self, model_path: str):
        super().__init__(
            f"Model not found at '{model_path}'. "
            "Run the training pipeline first or point --model-path to a valid directory.",
            details={"model_path": model_path},
        )


class ModelLoadError(ModelError):
    """Model files exist but could not be loaded (corrupt weights, version mismatch, etc.)."""

    def __init__(self, model_path: str, reason: str):
        super().__init__(
            f"Failed to load model from '{model_path}': {reason}",
            details={"model_path": model_path, "reason": reason},
        )


class ModelInferenceError(ModelError):
    """An error occurred during a forward pass."""

    def __init__(self, reason: str):
        super().__init__(
            f"Inference failed: {reason}",
            details={"reason": reason},
        )


# ---------------------------------------------------------------------------
# Input errors
# ---------------------------------------------------------------------------

class InputError(PIIDetectionError):
    """Base for input validation failures."""


class EmptyInputError(InputError):
    """Received an empty string, empty list, or blank file."""

    def __init__(self, context: str = "input"):
        super().__init__(
            f"Empty {context} provided. At least one non-blank text is required.",
            details={"context": context},
        )


class InputTooLargeError(InputError):
    """A single text exceeds the maximum allowed length."""

    def __init__(self, length: int, max_length: int):
        super().__init__(
            f"Input length {length} characters exceeds maximum allowed {max_length}.",
            details={"length": length, "max_length": max_length},
        )


class InvalidInputTypeError(InputError):
    """Received a value that is not a string where a string is expected."""

    def __init__(self, received_type: str, position: int = None):
        location = f" at position {position}" if position is not None else ""
        super().__init__(
            f"Expected a string{location}, got '{received_type}'.",
            details={"received_type": received_type, "position": position},
        )


# ---------------------------------------------------------------------------
# File parsing errors
# ---------------------------------------------------------------------------

class FileParsingError(PIIDetectionError):
    """Base for file parsing failures."""


class UnsupportedFileTypeError(FileParsingError):
    """File extension is not supported."""

    SUPPORTED = [".txt", ".csv", ".xlsx", ".xls"]

    def __init__(self, filename: str, extension: str):
        super().__init__(
            f"Unsupported file type '{extension}' for file '{filename}'. "
            f"Supported types: {', '.join(UnsupportedFileTypeError.SUPPORTED)}.",
            details={"filename": filename, "extension": extension,
                     "supported": UnsupportedFileTypeError.SUPPORTED},
        )


class FileReadError(FileParsingError):
    """File could not be opened or read (permissions, corrupt bytes, etc.)."""

    def __init__(self, filename: str, reason: str):
        super().__init__(
            f"Could not read file '{filename}': {reason}",
            details={"filename": filename, "reason": reason},
        )


class TextFileParsingError(FileParsingError):
    """Error while parsing a plain-text file into lines/chunks."""

    def __init__(self, filename: str, reason: str):
        super().__init__(
            f"Failed to parse text file '{filename}': {reason}",
            details={"filename": filename, "reason": reason},
        )


class CSVParsingError(FileParsingError):
    """Error while parsing a CSV file."""

    def __init__(self, filename: str, reason: str):
        super().__init__(
            f"Failed to parse CSV file '{filename}': {reason}",
            details={"filename": filename, "reason": reason},
        )


class ExcelParsingError(FileParsingError):
    """Error while parsing an Excel file."""

    def __init__(self, filename: str, reason: str):
        super().__init__(
            f"Failed to parse Excel file '{filename}': {reason}",
            details={"filename": filename, "reason": reason},
        )


# ---------------------------------------------------------------------------
# API errors
# ---------------------------------------------------------------------------

class APIError(PIIDetectionError):
    """Base for API request validation failures."""


class MissingFieldError(APIError):
    """A required field is absent from the request body."""

    def __init__(self, field: str):
        super().__init__(
            f"Missing required field '{field}' in request body.",
            details={"field": field},
        )


class InvalidFieldError(APIError):
    """A field is present but has the wrong type or an invalid value."""

    def __init__(self, field: str, reason: str):
        super().__init__(
            f"Invalid value for field '{field}': {reason}",
            details={"field": field, "reason": reason},
        )