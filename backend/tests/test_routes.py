"""Tests for the /api/form and /api/fill routes."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models import Confidence, ExtractedField, FillResponse

client = TestClient(app)


# ---------------------------------------------------------------------------
# GET /api/form/{bank_id}
# ---------------------------------------------------------------------------

class TestGetFormConfig:
    """Tests for the form config endpoint."""

    def test_get_demo_bank_config(self) -> None:
        """Should return the demo_bank form config."""
        response = client.get("/api/form/demo_bank")
        assert response.status_code == 200
        data = response.json()
        assert data["bank_id"] == "demo_bank"
        assert len(data["fields"]) == 13

    def test_get_nonexistent_bank_returns_404(self) -> None:
        """Should return 404 for an unknown bank_id."""
        response = client.get("/api/form/nonexistent")
        assert response.status_code == 404

    def test_list_available_forms(self) -> None:
        """GET /api/form should list available bank IDs."""
        response = client.get("/api/form")
        assert response.status_code == 200
        data = response.json()
        assert "demo_bank" in data


# ---------------------------------------------------------------------------
# POST /api/fill
# ---------------------------------------------------------------------------

class TestFillEndpoint:
    """Tests for the audio fill endpoint (Gemini mocked)."""

    @pytest.fixture
    def mock_fill_response(self) -> FillResponse:
        """Return a mock FillResponse."""
        return FillResponse(
            transcript="என் பெயர் ராஜா குமார்",
            language_detected="ta",
            fields={
                "full_name": ExtractedField(
                    value="ராஜா குமார்",
                    confidence=Confidence.HIGH,
                ),
                "mobile_number": ExtractedField(
                    value="9876543210",
                    confidence=Confidence.HIGH,
                ),
            },
        )

    def test_fill_rejects_invalid_language(self) -> None:
        """Should reject a language that isn't en or ta."""
        # Create a minimal WAV-like payload
        wav_bytes = b"RIFF" + b"\x00" * 100
        response = client.post(
            "/api/fill",
            data={"language": "fr", "bank_id": "demo_bank"},
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert response.status_code == 400
        assert "Language" in response.json()["detail"]

    def test_fill_rejects_empty_audio(self) -> None:
        """Should reject an empty audio file."""
        response = client.post(
            "/api/fill",
            data={"language": "en", "bank_id": "demo_bank"},
            files={"audio": ("test.wav", b"", "audio/wav")},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_fill_rejects_unknown_bank(self) -> None:
        """Should reject an unknown bank_id."""
        wav_bytes = b"RIFF" + b"\x00" * 100
        response = client.post(
            "/api/fill",
            data={"language": "en", "bank_id": "nonexistent"},
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert response.status_code == 400

    @patch("backend.app.routes.fill.extract_fields_from_audio")
    def test_fill_success_with_mocked_gemini(
        self,
        mock_extract: AsyncMock,
        mock_fill_response: FillResponse,
    ) -> None:
        """Should return extracted fields when Gemini call succeeds (mocked)."""
        mock_extract.return_value = mock_fill_response

        wav_bytes = b"RIFF" + b"\x00" * 100
        response = client.post(
            "/api/fill",
            data={"language": "ta", "bank_id": "demo_bank"},
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == "என் பெயர் ராஜா குமார்"
        assert data["fields"]["full_name"]["value"] == "ராஜா குமார்"
        assert data["fields"]["full_name"]["confidence"] == "high"
        mock_extract.assert_called_once()

    @patch("backend.app.routes.fill.extract_fields_from_audio")
    def test_fill_handles_gemini_failure_gracefully(
        self,
        mock_extract: AsyncMock,
    ) -> None:
        """Should return 500 with clean message when Gemini fails."""
        mock_extract.side_effect = RuntimeError("Gemini is down")

        wav_bytes = b"RIFF" + b"\x00" * 100
        response = client.post(
            "/api/fill",
            data={"language": "en", "bank_id": "demo_bank"},
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    """Tests for the health check endpoint."""

    def test_health_check(self) -> None:
        """Health check should return ok."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
