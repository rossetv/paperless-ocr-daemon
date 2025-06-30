FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    requests \
    pdf2image \
    Pillow \
    openai

# Entrypoint (script will be mounted)
CMD ["python3", "/app/ocr_daemon.py"]

