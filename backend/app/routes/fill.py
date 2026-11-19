"""Route: /api/fill — audio-to-form extraction using Gemini."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from ..ai import extract_fields_from_audio
from ..config import get_settings
from ..form_loader import FormLoadError, load_form_config
from ..models import ErrorResponse, FillResponse, FormConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["fill"])

_ALLOWED_AUDIO_TYPES: frozenset[str] = frozenset({
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "application/octet-stream",
})


def _validate_language(language: str) -> None:
    """Validate requested language code.

    Args:
        language: Language identifier string.

    Raises:
        HTTPException: If the language is not 'en' or 'ta'.
    """
    if language not in ("en", "ta"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Language must be 'en' or 'ta'.",
        )


def _validate_audio(audio: UploadFile, audio_bytes: bytes, max_bytes: int) -> None:
    """Validate audio upload content type and size constraints.

    Args:
        audio: Uploaded file object.
        audio_bytes: In-memory raw audio bytes.
        max_bytes: Maximum allowed byte length.

    Raises:
        HTTPException: If the format is unsupported, size exceeds limit, or file is empty.
    """
    if audio.content_type and audio.content_type not in _ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format: {audio.content_type}. Use WAV.",
        )

    if len(audio_bytes) > max_bytes:
        limit_mb = max_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file too large. Maximum is {limit_mb} MB.",
        )

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is empty.",
        )


def _load_form_or_400(bank_id: str, forms_dir: str) -> FormConfig:
    """Load bank form configuration or raise HTTP 400.

    Args:
        bank_id: Bank identifier.
        forms_dir: Directory containing form JSON files.

    Returns:
        Loaded FormConfig object.

    Raises:
        HTTPException: If the form configuration cannot be loaded.
    """
    try:
        return load_form_config(bank_id, forms_dir)
    except FormLoadError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


async def _extract_fields_or_500(
    audio_bytes: bytes,
    language: str,
    form_config: FormConfig,
) -> FillResponse:
    """Execute AI extraction or raise HTTP 500 on failure.

    Args:
        audio_bytes: Raw audio data.
        language: Language code.
        form_config: Loaded FormConfig.

    Returns:
        Extracted FillResponse.

    Raises:
        HTTPException: If Gemini processing fails.
    """
    try:
        return await extract_fields_from_audio(
            audio_bytes=audio_bytes,
            language=language,
            form_config=form_config,
        )
    except RuntimeError as exc:
        logger.error("AI extraction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Voice processing failed. Please try again.",
        ) from exc


@router.post(
    "/fill",
    response_model=FillResponse,
    summary="Extract form fields from spoken audio",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        413: {"model": ErrorResponse, "description": "Audio file too large"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "AI processing error"},
    },
)
async def fill_form(
    audio: Annotated[UploadFile, File(description="WAV audio file (16-bit PCM)")],
    language: Annotated[str, Form(description="Language code: 'en' or 'ta'")],
    bank_id: Annotated[str, Form(description="Bank form ID, e.g. 'demo_bank'")] = "demo_bank",
) -> FillResponse:
    """Accept an audio recording and extract form field values using Gemini.

    The endpoint sends the audio to Gemini along with the form field definitions
    and returns a transcript plus structured field values with confidence levels.
    Fields the customer did not mention are set to null.
    """
    settings = get_settings()

    _validate_language(language)
    audio_bytes = await audio.read()
    _validate_audio(audio, audio_bytes, settings.max_upload_bytes)

    form_config = _load_form_or_400(bank_id, settings.forms_dir)
    return await _extract_fields_or_500(audio_bytes, language, form_config)
