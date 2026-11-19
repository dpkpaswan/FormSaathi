"""Gemini AI integration for FormSaathi.

Handles audio transcription + field extraction and field explanations.
All Gemini calls have timeout, retry (max 2), and clean error handling.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from google import genai
from google.genai import types

from .config import get_settings
from .models import Confidence, ExtractedField, FillResponse, FormConfig, FormField
from .prompts import (
    EXPLAIN_SYSTEM_PROMPT,
    FILL_SYSTEM_PROMPT,
    build_explain_prompt,
    build_fill_user_prompt,
    format_field_descriptions,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RETRIES: int = 2
_TIMEOUT_SECONDS: float = 60.0


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

def _get_client() -> genai.Client:
    """Create a Gemini client with the configured API key."""
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Please set it in your .env file."
        )
    return genai.Client(api_key=settings.gemini_api_key)


def _get_model_name() -> str:
    """Return the configured Gemini model name."""
    return get_settings().gemini_model


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

async def _execute_gemini_attempt(
    client: genai.Client,
    model: str,
    contents: list[Any],
    config: types.GenerateContentConfig,
) -> Any:
    """Execute a single attempt to generate content via Gemini with timeout.

    Args:
        client: Active Gemini client instance.
        model: Gemini model identifier.
        contents: Input contents (audio, text prompts).
        config: Generation options and parameters.

    Returns:
        The response object from Gemini.
    """
    return await asyncio.wait_for(
        asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=contents,
            config=config,
        ),
        timeout=_TIMEOUT_SECONDS,
    )


def _handle_attempt_error(exc: Exception, attempt: int) -> Exception:
    """Log and return formatted error for a failed Gemini request attempt.

    Args:
        exc: Caught exception during attempt.
        attempt: Attempt counter (1-indexed).

    Returns:
        Normalized exception instance.
    """
    if isinstance(exc, TimeoutError):
        err = TimeoutError(
            f"Gemini request timed out after {_TIMEOUT_SECONDS}s "
            f"(attempt {attempt}/{_MAX_RETRIES})"
        )
        logger.warning("Gemini timeout on attempt %d/%d", attempt, _MAX_RETRIES)
        return err
    logger.warning("Gemini error on attempt %d/%d: %s", attempt, _MAX_RETRIES, exc)
    return exc


async def _call_gemini_with_retry(
    client: genai.Client,
    model: str,
    contents: list[Any],
    config: types.GenerateContentConfig,
) -> Any:
    """Call Gemini with timeout and retry logic.

    Args:
        client: The Gemini client instance.
        model: Model name string.
        contents: Content parts for the request.
        config: Generation configuration.

    Returns:
        The Gemini response object.

    Raises:
        RuntimeError: If all retries fail or timeout occurs.
    """
    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return await _execute_gemini_attempt(client, model, contents, config)
        except Exception as exc:
            last_error = _handle_attempt_error(exc, attempt)

        if attempt < _MAX_RETRIES:
            await asyncio.sleep(1.0 * attempt)

    raise RuntimeError(
        f"Gemini call failed after {_MAX_RETRIES} attempts: {last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# Build the response schema for structured output
# ---------------------------------------------------------------------------

def _build_field_schema(field_label: str) -> dict[str, Any]:
    """Build the JSON schema definition for a single form field extraction.

    Args:
        field_label: Human-readable English field label.

    Returns:
        Schema dictionary for the field.
    """
    return {
        "type": "object",
        "properties": {
            "value": {
                "type": "string",
                "nullable": True,
                "description": f"Extracted value for {field_label}, or null if not mentioned.",
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Confidence level in the extraction.",
            },
        },
        "required": ["value", "confidence"],
    }


def _build_fill_response_schema(form_config: FormConfig) -> dict[str, Any]:
    """Build a JSON schema for Gemini structured output based on form fields.

    Args:
        form_config: The bank form configuration.

    Returns:
        A JSON schema dict for the expected response.
    """
    field_properties = {
        field.id: _build_field_schema(field.label_en)
        for field in form_config.fields
    }

    return {
        "type": "object",
        "properties": {
            "transcript": {
                "type": "string",
                "description": "Full transcription of the spoken audio.",
            },
            "language_detected": {
                "type": "string",
                "enum": ["en", "ta"],
                "description": "Detected language.",
            },
            "fields": {
                "type": "object",
                "properties": field_properties,
                "required": [f.id for f in form_config.fields],
            },
        },
        "required": ["transcript", "language_detected", "fields"],
    }


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

def _prepare_fill_prompt(form_config: FormConfig, language: str) -> str:
    """Format form field definitions and build the extraction user prompt.

    Args:
        form_config: Bank form configuration.
        language: Language code ('en' or 'ta').

    Returns:
        Formatted user prompt for Gemini.
    """
    field_dicts = [
        {
            "id": f.id,
            "label_en": f.label_en,
            "label_ta": f.label_ta,
            "type": f.type,
            "options": [o.model_dump() for o in f.options] if f.options else None,
            "required": f.required,
        }
        for f in form_config.fields
    ]
    field_desc = format_field_descriptions(field_dicts)
    return build_fill_user_prompt(language, field_desc)


def _build_fill_generation_config(
    form_config: FormConfig,
) -> types.GenerateContentConfig:
    """Build generation config with structured schema for form extraction.

    Args:
        form_config: Bank form configuration.

    Returns:
        Configured GenerateContentConfig instance.
    """
    return types.GenerateContentConfig(
        system_instruction=FILL_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=_build_fill_response_schema(form_config),
        temperature=0.1,
    )


def _parse_gemini_json_response(raw_text: str) -> dict[str, Any]:
    """Parse JSON string returned from Gemini structured response.

    Args:
        raw_text: Raw JSON string output from Gemini.

    Returns:
        Parsed dictionary.

    Raises:
        RuntimeError: If JSON parsing fails.
    """
    try:
        data: dict[str, Any] = json.loads(raw_text)
        return data
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse Gemini JSON response: %s", raw_text[:500])
        raise RuntimeError(
            "AI returned an invalid response. Please try again."
        ) from exc


def _map_extracted_fields(
    raw_fields: dict[str, Any],
    form_config: FormConfig,
) -> dict[str, ExtractedField]:
    """Map raw dictionary items to strongly-typed ExtractedField objects.

    Args:
        raw_fields: Raw dictionary of extracted fields from Gemini.
        form_config: The bank form configuration.

    Returns:
        Dictionary mapping field IDs to validated ExtractedField models.
    """
    extracted_fields: dict[str, ExtractedField] = {}
    for field in form_config.fields:
        raw = raw_fields.get(field.id, {})
        value = raw.get("value")
        conf_str = raw.get("confidence", "low")
        try:
            confidence = Confidence(conf_str)
        except ValueError:
            confidence = Confidence.LOW

        extracted_fields[field.id] = ExtractedField(
            value=value if value else None,
            confidence=confidence,
        )
    return extracted_fields


# ---------------------------------------------------------------------------
# Public: extract fields from audio
# ---------------------------------------------------------------------------

async def extract_fields_from_audio(
    audio_bytes: bytes,
    language: str,
    form_config: FormConfig,
) -> FillResponse:
    """Transcribe audio and extract form field values using Gemini.

    Args:
        audio_bytes: Raw WAV audio data.
        language: Preferred language code (``en`` or ``ta``).
        form_config: The bank form configuration.

    Returns:
        A ``FillResponse`` with transcript, detected language, and extracted fields.

    Raises:
        RuntimeError: If the Gemini call fails after retries.
    """
    client = _get_client()
    user_prompt = _prepare_fill_prompt(form_config, language)
    contents = [
        types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
        types.Part.from_text(text=user_prompt),
    ]
    config = _build_fill_generation_config(form_config)

    response = await _call_gemini_with_retry(client, _get_model_name(), contents, config)
    data = _parse_gemini_json_response(response.text)

    return FillResponse(
        transcript=data.get("transcript", ""),
        fields=_map_extracted_fields(data.get("fields", {}), form_config),
        language_detected=data.get("language_detected", language),
    )


# ---------------------------------------------------------------------------
# Public: explain a field
# ---------------------------------------------------------------------------

def _get_field_by_id(form_config: FormConfig, field_id: str) -> FormField:
    """Retrieve a FormField by its identifier or raise ValueError.

    Args:
        form_config: Bank form configuration.
        field_id: Field identifier to find.

    Returns:
        Matching FormField.

    Raises:
        ValueError: If field_id is not found in form_config.
    """
    field = next((f for f in form_config.fields if f.id == field_id), None)
    if field is None:
        raise ValueError(f"Unknown field: {field_id}")
    return field


def _build_explain_user_prompt(
    field: FormField,
    language: str,
    user_question: str | None,
) -> str:
    """Build the user prompt for field explanation based on language and help text.

    Args:
        field: Target FormField.
        language: Language code ('en' or 'ta').
        user_question: Optional specific question asked by the user.

    Returns:
        Formatted user prompt.
    """
    label = field.label_ta if language == "ta" else field.label_en
    help_text = field.help_ta if language == "ta" else field.help_en
    return build_explain_prompt(label, help_text, language, user_question)


async def explain_field(
    field_id: str,
    language: str,
    form_config: FormConfig,
    user_question: str | None = None,
) -> str:
    """Generate a simple explanation of a form field using Gemini.

    Args:
        field_id: The form field identifier.
        language: Language code (``en`` or ``ta``).
        form_config: The bank form configuration.
        user_question: Optional spoken/typed question from the user.

    Returns:
        A simple explanation string in the requested language.

    Raises:
        ValueError: If the field_id is not found in the form config.
        RuntimeError: If the Gemini call fails after retries.
    """
    field = _get_field_by_id(form_config, field_id)
    user_prompt = _build_explain_user_prompt(field, language, user_question)

    config = types.GenerateContentConfig(
        system_instruction=EXPLAIN_SYSTEM_PROMPT,
        temperature=0.3,
        max_output_tokens=200,
    )

    response = await _call_gemini_with_retry(
        _get_client(), _get_model_name(), [user_prompt], config
    )
    return str(response.text).strip()
