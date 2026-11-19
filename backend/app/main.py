"""FormSaathi — FastAPI application entry point.

A voice-powered AI assistant for bank account opening forms.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .routes import fill, form

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FormSaathi API",
    description=(
        "AI voice assistant for bank account opening forms. "
        "Demo only — use fake data."
    ),
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

settings = get_settings()
origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handler — never leak stack traces
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled exceptions and return a clean error response."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(form.router)
app.include_router(fill.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok", "service": "FormSaathi"}
