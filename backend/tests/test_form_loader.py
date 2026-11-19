"""Tests for form configuration loading."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from backend.app.form_loader import FormLoadError, load_form_config, list_available_banks
from backend.app.models import FormConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_form_data() -> dict:
    """Return a minimal valid form config dict."""
    return {
        "bank_id": "test_bank",
        "bank_name_en": "Test Bank",
        "bank_name_ta": "சோதனை வங்கி",
        "form_title_en": "Test Form",
        "form_title_ta": "சோதனை படிவம்",
        "fields": [
            {
                "id": "full_name",
                "label_en": "Full Name",
                "label_ta": "முழு பெயர்",
                "type": "text",
                "required": True,
                "options": None,
                "help_en": "Your full name.",
                "help_ta": "உங்கள் முழு பெயர்.",
                "validation": {"rule": "name", "min_length": 2, "max_length": 100},
            }
        ],
    }


@pytest.fixture
def temp_forms_dir(sample_form_data: dict) -> Path:
    """Create a temporary forms directory with a test form JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        form_file = path / "test_bank.json"
        form_file.write_text(json.dumps(sample_form_data), encoding="utf-8")
        yield path


# ---------------------------------------------------------------------------
# Tests: load_form_config
# ---------------------------------------------------------------------------

class TestLoadFormConfig:
    """Tests for the form config loader."""

    def test_loads_valid_config(self, temp_forms_dir: Path) -> None:
        """A valid JSON file should load and parse correctly."""
        config = load_form_config("test_bank", str(temp_forms_dir))
        assert isinstance(config, FormConfig)
        assert config.bank_id == "test_bank"
        assert config.bank_name_en == "Test Bank"
        assert len(config.fields) == 1
        assert config.fields[0].id == "full_name"

    def test_missing_bank_raises_error(self, temp_forms_dir: Path) -> None:
        """A non-existent bank_id should raise FormLoadError."""
        with pytest.raises(FormLoadError, match="Form config not found"):
            load_form_config("nonexistent_bank", str(temp_forms_dir))

    def test_invalid_json_raises_error(self, temp_forms_dir: Path) -> None:
        """A malformed JSON file should raise FormLoadError."""
        bad_file = temp_forms_dir / "bad_bank.json"
        bad_file.write_text("not valid json {{{", encoding="utf-8")
        with pytest.raises(FormLoadError, match="Cannot read"):
            load_form_config("bad_bank", str(temp_forms_dir))

    def test_missing_required_field_raises_error(self, temp_forms_dir: Path) -> None:
        """A JSON file missing required fields should raise FormLoadError."""
        incomplete = temp_forms_dir / "incomplete_bank.json"
        incomplete.write_text(
            json.dumps({"bank_id": "incomplete_bank"}),
            encoding="utf-8",
        )
        with pytest.raises(FormLoadError, match="Invalid form config"):
            load_form_config("incomplete_bank", str(temp_forms_dir))

    def test_bank_id_mismatch_raises_error(self, temp_forms_dir: Path) -> None:
        """If file bank_id doesn't match requested ID, raise FormLoadError."""
        mismatch = temp_forms_dir / "wrong_bank.json"
        data = {
            "bank_id": "actually_different",
            "bank_name_en": "X",
            "bank_name_ta": "X",
            "form_title_en": "X",
            "form_title_ta": "X",
            "fields": [],
        }
        mismatch.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(FormLoadError, match="bank_id mismatch"):
            load_form_config("wrong_bank", str(temp_forms_dir))


# ---------------------------------------------------------------------------
# Tests: load the actual demo_bank.json
# ---------------------------------------------------------------------------

class TestDemoBankConfig:
    """Tests for the real demo_bank.json form config."""

    def test_demo_bank_loads(self) -> None:
        """The demo_bank.json file should load successfully."""
        config = load_form_config("demo_bank")
        assert config.bank_id == "demo_bank"
        assert len(config.fields) == 13

    def test_demo_bank_has_required_fields(self) -> None:
        """The demo_bank form should have the expected required fields."""
        config = load_form_config("demo_bank")
        required_ids = {f.id for f in config.fields if f.required}
        assert "full_name" in required_ids
        assert "date_of_birth" in required_ids
        assert "mobile_number" in required_ids
        assert "id_proof_type" in required_ids

    def test_demo_bank_has_no_id_number_field(self) -> None:
        """The form must NOT have any field for Aadhaar/PAN numbers."""
        config = load_form_config("demo_bank")
        field_ids = {f.id for f in config.fields}
        forbidden = {"aadhaar_number", "pan_number", "id_number", "voter_id_number"}
        assert field_ids.isdisjoint(forbidden), (
            f"Found forbidden ID number fields: {field_ids & forbidden}"
        )

    def test_demo_bank_id_proof_is_choice(self) -> None:
        """id_proof_type should be a choice field, not collecting numbers."""
        config = load_form_config("demo_bank")
        id_field = next(f for f in config.fields if f.id == "id_proof_type")
        assert id_field.type == "choice"
        assert id_field.options is not None
        option_values = [o.value for o in id_field.options]
        assert "aadhaar" in option_values
        assert "voter_id" in option_values


# ---------------------------------------------------------------------------
# Tests: list_available_banks
# ---------------------------------------------------------------------------

class TestListAvailableBanks:
    """Tests for listing available bank form IDs."""

    def test_lists_banks_in_temp_dir(self, temp_forms_dir: Path) -> None:
        """Should find the test bank in a temp directory."""
        banks = list_available_banks(str(temp_forms_dir))
        assert "test_bank" in banks

    def test_lists_demo_bank(self) -> None:
        """Should find demo_bank in the default forms directory."""
        banks = list_available_banks()
        assert "demo_bank" in banks

    def test_empty_dir_returns_empty_list(self) -> None:
        """An empty directory should return an empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            banks = list_available_banks(tmpdir)
            assert banks == []
