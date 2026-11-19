"""Route: /api/form/{bank_id} — return form field configuration."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from ..config import get_settings
from ..form_loader import FormLoadError, list_available_banks, load_form_config
from ..models import FormConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["form"])


@router.get(
    "/form/{bank_id}",
    response_model=FormConfig,
    summary="Get form field configuration for a bank",
    responses={
        404: {"description": "Bank form not found"},
    },
)
async def get_form_config(bank_id: str) -> FormConfig:
    """Return the form field configuration for the specified bank.

    The frontend uses this to render the form dynamically.
    Adding a new bank only requires a new JSON file in the ``forms/`` directory.
    """
    settings = get_settings()
    try:
        return load_form_config(bank_id, settings.forms_dir)
    except FormLoadError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get(
    "/form",
    response_model=list[str],
    summary="List available bank form IDs",
)
async def list_forms() -> list[str]:
    """Return a list of all available bank form IDs."""
    settings = get_settings()
    return list_available_banks(settings.forms_dir)
