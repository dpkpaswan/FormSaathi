"""Pydantic models for FormSaathi API requests and responses."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Form configuration models (mirror the JSON schema)
# ---------------------------------------------------------------------------

class ValidationRule(BaseModel):
    """Validation configuration for a single form field."""

    rule: str
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    format: str | None = None
    min_age: int | None = None
    max_age: int | None = None
    allowed: list[str] | None = None


class FieldOption(BaseModel):
    """An option for select / choice fields."""

    value: str
    label_en: str
    label_ta: str


class FormField(BaseModel):
    """A single field in a bank account opening form."""

    id: str
    label_en: str
    label_ta: str
    type: str  # text | date | phone | select | choice
    required: bool
    options: list[FieldOption] | None = None
    help_en: str
    help_ta: str
    validation: ValidationRule


class FormConfig(BaseModel):
    """Full form configuration loaded from a bank JSON file."""

    bank_id: str
    bank_name_en: str
    bank_name_ta: str
    form_title_en: str
    form_title_ta: str
    fields: list[FormField]


# ---------------------------------------------------------------------------
# Confidence enum
# ---------------------------------------------------------------------------

class Confidence(str, Enum):
    """AI extraction confidence level for a single field."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# /api/fill
# ---------------------------------------------------------------------------

class ExtractedField(BaseModel):
    """A single extracted field value with confidence."""

    value: str | None = Field(
        default=None,
        description="Extracted value, or null if not mentioned.",
    )
    confidence: Confidence = Field(
        default=Confidence.LOW,
        description="AI confidence in this extraction.",
    )


class FillResponse(BaseModel):
    """Response from the /api/fill endpoint."""

    transcript: str = Field(description="Transcribed text from the audio.")
    fields: dict[str, ExtractedField] = Field(
        description="Extracted form field values keyed by field id.",
    )
    language_detected: str = Field(
        default="en",
        description="Detected language code (en or ta).",
    )


# ---------------------------------------------------------------------------
# Generic error / status responses
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    """A single validation or processing error."""

    field_id: str | None = None
    message_en: str
    message_ta: str


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str
    errors: list[ErrorDetail] | None = None
