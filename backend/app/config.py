"""FormSaathi backend configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings read from environment variables or .env file."""

    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key (required for AI features).",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model name to use for inference.",
    )
    manager_password: str = Field(
        default="changeme",
        description="Password to access the manager dashboard.",
    )
    host: str = Field(default="0.0.0.0", description="Server bind host.")
    port: int = Field(default=8000, description="Server bind port.")
    max_upload_bytes: int = Field(
        default=5 * 1024 * 1024,
        description="Maximum upload size in bytes (default 5 MB).",
    )
    max_audio_seconds: int = Field(
        default=30,
        description="Maximum audio duration in seconds.",
    )
    db_path: str = Field(
        default="formsaathi.db",
        description="SQLite database file path.",
    )
    forms_dir: str = Field(
        default="forms",
        description="Directory containing bank form JSON configs.",
    )
    pdf_dir: str = Field(
        default="pdfs",
        description="Directory for generated PDF files.",
    )
    cors_origins: str = Field(
        default="http://localhost:5173",
        description="Comma-separated list of allowed CORS origins.",
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
