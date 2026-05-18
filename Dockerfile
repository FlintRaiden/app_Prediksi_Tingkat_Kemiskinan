# ── Hugging Face Spaces — Docker SDK ────────────────────────────────────────
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

# Hugging Face Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /workspace
USER 1000

# Expose port expected by HF Spaces
EXPOSE 7860

# Launch via Gunicorn — chdir into app/ so Flask finds templates & static
CMD ["gunicorn", \
     "--chdir", "app", \
     "app:app", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "1", \
     "--timeout", "300", \
     "--log-level", "info", \
     "--access-logfile", "-"]
