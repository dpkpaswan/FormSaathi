# ===========================================================================
# FormSaathi — Multi-stage Dockerfile
# Stage 1: Build React frontend
# Stage 2: Python runtime with FastAPI serving the built frontend
# ===========================================================================

# --- Stage 1: Build frontend ---
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend

# Install dependencies first (layer caching)
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund 2>/dev/null || echo "No package.json yet — using placeholder frontend"

# Copy source and build
COPY frontend/ ./
RUN if [ -f "vite.config.ts" ]; then npm run build; else echo "Skipping frontend build — using pre-built dist"; fi


# --- Stage 2: Python runtime ---
FROM python:3.12-slim AS runtime

# Security: run as non-root
RUN groupadd -r formsaathi && useradd -r -g formsaathi -m formsaathi

WORKDIR /app

# Install Python dependencies (layer caching)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend source
COPY backend/ ./backend/

# Copy form configs
COPY forms/ ./forms/

# Copy built frontend from stage 1
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Create directories for runtime data
RUN mkdir -p pdfs && chown -R formsaathi:formsaathi /app

# Switch to non-root user
USER formsaathi

# Environment defaults
ENV HOST=0.0.0.0 \
    PORT=8000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Run with uvicorn
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
