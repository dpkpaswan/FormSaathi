"""Route: /api/fill — audio-to-form extraction using Gemini."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from ..ai import extract_fields_from_audio
from ..config import get_settings
from ..form_loader import FormLoadError, load_form_config
from ..models import FillResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["fill"])


@router.post(
    "/fill",
    response_model=FillResponse,
    summary="Extract form fields from spoken audio",
    responses={
        400: {"description": "Invalid input"},
        413: {"description": "Audio file too large"},
        422: {"description": "Validation error"},
        500: {"description": "AI processing error"},
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

    # Validate language
    if language not in ("en", "ta"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Language must be 'en' or 'ta'.",
        )

    # Validate content type
    if audio.content_type and audio.content_type not in (
        "audio/wav",
        "audio/wave",
        "audio/x-wav",
        "application/octet-stream",
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format: {audio.content_type}. Use WAV.",
        )

    # Read and validate size
    audio_bytes = await audio.read()
    if len(audio_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file too large. Maximum is {settings.max_upload_bytes // (1024 * 1024)} MB.",
        )

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is empty.",
        )

    # Load form config
    try:
        form_config = load_form_config(bank_id, settings.forms_dir)
    except FormLoadError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    # Call Gemini
    try:
        result = await extract_fields_from_audio(
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

    return result
