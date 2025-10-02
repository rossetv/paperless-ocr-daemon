# Stage 1: Builder and Tester
FROM python:3.11-slim as builder

# Install system dependencies required for building and running tests
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy project definition and test requirements
COPY pyproject.toml requirements-dev.txt ./

# Install testing dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy the rest of the project
COPY src/ ./src/
COPY tests/ ./tests/

# Install the project and its dependencies
RUN pip install --no-cache-dir .

# Run tests
RUN pytest

# Stage 2: Final Production Image
FROM python:3.11-slim

# Install only essential runtime system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application source code
COPY --from=builder /app/src ./src

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Set the entrypoint to run the refactored application
CMD ["python3", "-m", "src.paperless_ocr.main"]