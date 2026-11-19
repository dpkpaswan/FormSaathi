"""Gemini AI integration for FormSaathi.

Handles audio transcription + field extraction and field explanations.
All Gemini calls have timeout, retry (max 2), and clean error handling.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

from google import genai
from google.genai import types

from .config import get_settings
from .models import Confidence, ExtractedField, FillResponse, FormConfig
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

_MAX_RETRIES = 2
_TIMEOUT_SECONDS = 60


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
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=config,
                ),
                timeout=_TIMEOUT_SECONDS,
            )
            return response
        except asyncio.TimeoutError:
            last_error = TimeoutError(
                f"Gemini request timed out after {_TIMEOUT_SECONDS}s "
                f"(attempt {attempt}/{_MAX_RETRIES})"
            )
            logger.warning("Gemini timeout on attempt %d/%d", attempt, _MAX_RETRIES)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Gemini error on attempt %d/%d: %s", attempt, _MAX_RETRIES, exc
            )

        if attempt < _MAX_RETRIES:
            await asyncio.sleep(1.0 * attempt)  # Simple backoff

    raise RuntimeError(
        f"Gemini call failed after {_MAX_RETRIES} attempts: {last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# Build the response schema for structured output
# ---------------------------------------------------------------------------

def _build_fill_response_schema(form_config: FormConfig) -> dict[str, Any]:
    """Build a JSON schema for Gemini structured output based on form fields.

    Args:
        form_config: The bank form configuration.

    Returns:
        A JSON schema dict for the expected response.
    """
    field_properties: dict[str, Any] = {}
    for field in form_config.fields:
        field_properties[field.id] = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "nullable": True,
                    "description": f"Extracted value for {field.label_en}, or null if not mentioned.",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Confidence level in the extraction.",
                },
            },
            "required": ["value", "confidence"],
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
    model = _get_model_name()

    # Build field descriptions for the prompt
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
    user_prompt = build_fill_user_prompt(language, field_desc)

    # Build contents with audio
    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type="audio/wav",
    )
    text_part = types.Part.from_text(text=user_prompt)

    response_schema = _build_fill_response_schema(form_config)

    config = types.GenerateContentConfig(
        system_instruction=FILL_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=response_schema,
        temperature=0.1,  # Low temperature for extraction accuracy
    )

    response = await _call_gemini_with_retry(
        client=client,
        model=model,
        contents=[audio_part, text_part],
        config=config,
    )

    # Parse structured response
    raw_text = response.text
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse Gemini JSON response: %s", raw_text[:500])
        raise RuntimeError(
            "AI returned an invalid response. Please try again."
        ) from exc

    # Build FillResponse
    extracted_fields: dict[str, ExtractedField] = {}
    raw_fields = data.get("fields", {})
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

    return FillResponse(
        transcript=data.get("transcript", ""),
        fields=extracted_fields,
        language_detected=data.get("language_detected", language),
    )


# ---------------------------------------------------------------------------
# Public: explain a field
# ---------------------------------------------------------------------------

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
    # Find the field
    field = None
    for f in form_config.fields:
        if f.id == field_id:
            field = f
            break

    if field is None:
        raise ValueError(f"Unknown field: {field_id}")

    client = _get_client()
    model = _get_model_name()

    label = field.label_ta if language == "ta" else field.label_en
    help_text = field.help_ta if language == "ta" else field.help_en

    user_prompt = build_explain_prompt(label, help_text, language, user_question)

    config = types.GenerateContentConfig(
        system_instruction=EXPLAIN_SYSTEM_PROMPT,
        temperature=0.3,
        max_output_tokens=200,
    )

    response = await _call_gemini_with_retry(
        client=client,
        model=model,
        contents=[user_prompt],
        config=config,
    )

    return response.text.strip()
