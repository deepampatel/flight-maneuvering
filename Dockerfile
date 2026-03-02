# ============================================
# Intercept - Multi-stage Docker build
# ============================================
# Stage 1: Build frontend
# Stage 2: Run backend + serve static frontend
# ============================================

# --- Stage 1: Build Frontend ---
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --production=false
COPY frontend/ ./

# Google Maps API key (optional — injected at build time for Vite)
ARG VITE_GOOGLE_MAPS_API_KEY=""
ENV VITE_GOOGLE_MAPS_API_KEY=$VITE_GOOGLE_MAPS_API_KEY

RUN npm run build

# --- Stage 2: Python Backend ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install Python dependencies
COPY backend/pyproject.toml ./backend/
RUN pip install --no-cache-dir \
    fastapi>=0.109.0 \
    "uvicorn[standard]>=0.27.0" \
    websockets>=12.0 \
    numpy>=1.26.0 \
    pydantic>=2.5.0 \
    scipy>=1.11.0

# Copy backend source
COPY backend/ ./backend/

# Copy built frontend
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Create recordings directory
RUN mkdir -p /app/backend/recordings

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/scenarios')" || exit 1

# Run
ENV PYTHONPATH=/app:/app/backend
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
