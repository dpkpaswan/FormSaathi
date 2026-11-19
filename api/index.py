"""Vercel Serverless Function entrypoint for FormSaathi FastAPI app."""

import sys
from pathlib import Path

# Add project root directory to sys.path so 'backend' and 'forms' are importable
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.app.main import app
