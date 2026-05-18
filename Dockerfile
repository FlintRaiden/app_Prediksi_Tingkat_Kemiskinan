# ── Railway Deployment ───────────────────────────────────────────────────────
# Python 3.11 slim base
FROM python:3.11-slim

# System dependencies (needed for TensorFlow & scientific libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /workspace

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose port (Railway uses dynamic $PORT, default fallback 8000)
EXPOSE 8000

# Launch via Gunicorn — chdir into app/ so Flask finds templates & static
CMD ["sh", "-c", "gunicorn --chdir app app:app --bind 0.0.0.0:${PORT:-8000} --workers 1 --timeout 300 --log-level info --access-logfile -"]
