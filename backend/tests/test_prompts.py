"""Tests for prompt template functions."""

from __future__ import annotations

from backend.app.prompts import (
    build_explain_prompt,
    build_fill_user_prompt,
    format_field_descriptions,
)


class TestFormatFieldDescriptions:
    """Tests for field description formatting."""

    def test_basic_field(self) -> None:
        """Should format a basic text field."""
        fields = [
            {
                "id": "full_name",
                "label_en": "Full Name",
                "label_ta": "முழு பெயர்",
                "type": "text",
                "options": None,
                "required": True,
            }
        ]
        result = format_field_descriptions(fields)
        assert "full_name" in result
        assert "Full Name" in result
        assert "[REQUIRED]" in result

    def test_select_field_with_options(self) -> None:
        """Should include options for select fields."""
        fields = [
            {
                "id": "gender",
                "label_en": "Gender",
                "label_ta": "பாலினம்",
                "type": "select",
                "options": [
                    {"value": "male", "label_en": "Male"},
                    {"value": "female", "label_en": "Female"},
                ],
                "required": True,
            }
        ]
        result = format_field_descriptions(fields)
        assert "male=Male" in result
        assert "female=Female" in result


class TestBuildFillUserPrompt:
    """Tests for the fill user prompt builder."""

    def test_english_prompt(self) -> None:
        """Should mention English for 'en' language."""
        prompt = build_fill_user_prompt("en", "field list here")
        assert "English" in prompt
        assert "field list here" in prompt

    def test_tamil_prompt(self) -> None:
        """Should mention Tamil for 'ta' language."""
        prompt = build_fill_user_prompt("ta", "field list here")
        assert "Tamil" in prompt


class TestBuildExplainPrompt:
    """Tests for the explain prompt builder."""

    def test_without_question(self) -> None:
        """Should build a basic explain prompt without user question."""
        prompt = build_explain_prompt("Full Name", "Your name", "en")
        assert "Full Name" in prompt
        assert "English" in prompt
        assert "Explain this field" in prompt

    def test_with_question(self) -> None:
        """Should include the user's question."""
        prompt = build_explain_prompt(
            "Nominee", "Person who gets money", "ta", "What is nominee?"
        )
        assert "Tamil" in prompt
        assert "What is nominee?" in prompt
