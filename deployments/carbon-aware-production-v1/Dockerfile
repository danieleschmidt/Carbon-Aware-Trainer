# Production Dockerfile for Carbon-Aware Trainer
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash carbon-trainer
USER carbon-trainer
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.prod.txt .
RUN pip install --no-cache-dir --user -r requirements.prod.txt

# Copy application code
COPY --chown=carbon-trainer:carbon-trainer src/ ./src/
COPY --chown=carbon-trainer:carbon-trainer examples/ ./examples/

# Create necessary directories
RUN mkdir -p /app/checkpoints /app/logs /app/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import carbon_aware_trainer; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "carbon_aware_trainer.cli"]
