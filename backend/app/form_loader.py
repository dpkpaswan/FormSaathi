"""Form configuration loader.

Reads bank form JSON files from the forms directory.  Adding a new bank
requires only dropping a new JSON file — no code changes needed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from functools import lru_cache

from .models import FormConfig

logger = logging.getLogger(__name__)


class FormLoadError(Exception):
    """Raised when a form config file cannot be loaded or parsed."""


def _resolve_forms_dir(forms_dir: str) -> Path:
    """Resolve the forms directory path relative to the project root."""
    path = Path(forms_dir)
    if path.is_absolute() and path.is_dir():
        return path
    if path.is_dir():
        return path.resolve()
    # Resolve relative to the project root (two levels up from this file)
    project_root = Path(__file__).resolve().parent.parent.parent
    candidate = project_root / forms_dir
    if candidate.is_dir():
        return candidate
    cwd_candidate = Path.cwd() / forms_dir
    if cwd_candidate.is_dir():
        return cwd_candidate
    return candidate


def load_form_config(bank_id: str, forms_dir: str = "forms") -> FormConfig:
    """Load and validate a bank form configuration from its JSON file.

    Args:
        bank_id: The bank identifier (e.g. ``demo_bank``).
        forms_dir: Directory containing form JSON files.

    Returns:
        Parsed and validated ``FormConfig`` object.

    Raises:
        FormLoadError: If the file is missing, unreadable, or invalid.
    """
    base_dir = _resolve_forms_dir(forms_dir)
    file_path = base_dir / f"{bank_id}.json"

    if not file_path.is_file():
        raise FormLoadError(
            f"Form config not found: {file_path}. "
            f"Available banks: {list_available_banks(forms_dir)}"
        )

    try:
        raw = file_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise FormLoadError(f"Cannot read form config {file_path}: {exc}") from exc

    try:
        config = FormConfig.model_validate(data)
    except Exception as exc:
        raise FormLoadError(
            f"Invalid form config in {file_path}: {exc}"
        ) from exc

    # Sanity check: bank_id in the file must match the requested one
    if config.bank_id != bank_id:
        raise FormLoadError(
            f"bank_id mismatch: file says '{config.bank_id}', "
            f"but requested '{bank_id}'."
        )

    logger.info("Loaded form config for bank '%s' with %d fields.", bank_id, len(config.fields))
    return config


def list_available_banks(forms_dir: str = "forms") -> list[str]:
    """List all available bank IDs from the forms directory.

    Returns:
        A list of bank_id strings (file stems of .json files).
    """
    base_dir = _resolve_forms_dir(forms_dir)
    if not base_dir.is_dir():
        return []
    return sorted(p.stem for p in base_dir.glob("*.json"))
