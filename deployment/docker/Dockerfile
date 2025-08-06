# Carbon-Aware-Trainer Production Dockerfile
# Multi-stage build for optimized production deployment

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=latest

# Add metadata labels
LABEL maintainer="Daniel Schmidt <daniel@terragonlabs.com>" \
      org.opencontainers.image.title="Carbon-Aware-Trainer" \
      org.opencontainers.image.description="Intelligent ML training scheduler for carbon reduction" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.url="https://github.com/danieleschmidt/carbon-aware-trainer" \
      org.opencontainers.image.source="https://github.com/danieleschmidt/carbon-aware-trainer"

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml setup.py ./
COPY src/ ./src/

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip wheel

# Install package in production mode
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir gunicorn uvicorn[standard]

# Production stage
FROM python:3.11-slim as production

# Set production build arguments
ARG BUILD_DATE
ARG VCS_REF  
ARG VERSION=latest

# Add metadata labels
LABEL maintainer="Daniel Schmidt <daniel@terragonlabs.com>" \
      org.opencontainers.image.title="Carbon-Aware-Trainer" \
      org.opencontainers.image.description="Intelligent ML training scheduler for carbon reduction" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="MIT"

# Create non-root user for security
RUN groupadd -r carbontrainer && useradd -r -g carbontrainer -u 1001 carbontrainer

# Install production system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CARBON_AWARE_ENVIRONMENT=production \
    CARBON_AWARE_LOG_LEVEL=info \
    CARBON_AWARE_WORKERS=2

# Create application directories
RUN mkdir -p /app/data /app/logs /app/config /app/tmp && \
    chown -R carbontrainer:carbontrainer /app

# Set working directory
WORKDIR /app

# Copy application files
COPY --chown=carbontrainer:carbontrainer src/ ./src/
COPY --chown=carbontrainer:carbontrainer deployment/config/ ./config/

# Create health check script
RUN echo '#!/bin/bash\npython -m carbon_aware_trainer.core.health --check' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh && \
    chown carbontrainer:carbontrainer /app/healthcheck.sh

# Switch to non-root user
USER carbontrainer

# Expose port for web interface (if implemented)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /app/healthcheck.sh || exit 1

# Default command
CMD ["python", "-m", "carbon_aware_trainer.cli", "--serve", "--host", "0.0.0.0", "--port", "8080"]

# Development stage (for development builds)
FROM python:3.11-slim as development

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Install in development mode with all extras
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev,all]"

# Set development environment
ENV CARBON_AWARE_ENVIRONMENT=development \
    CARBON_AWARE_LOG_LEVEL=debug \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Development command (override with docker run command)
CMD ["python", "-m", "carbon_aware_trainer.cli", "--help"]

# GPU-enabled stage (for CUDA support)
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as gpu

# Install Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Create non-root user
RUN groupadd -r carbontrainer && useradd -r -g carbontrainer carbontrainer

# Set working directory
WORKDIR /app

# Copy application
COPY --chown=carbontrainer:carbontrainer . .

# Install dependencies with PyTorch GPU support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -e ".[pytorch,monitoring]"

# Set GPU environment
ENV CARBON_AWARE_ENVIRONMENT=production \
    CARBON_AWARE_LOG_LEVEL=info \
    CUDA_VISIBLE_DEVICES=all

# Switch to non-root user
USER carbontrainer

# Expose port
EXPOSE 8080

# Health check with GPU support
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD python -c "import torch; torch.cuda.is_available()" && \
        python -m carbon_aware_trainer.core.health --check || exit 1

# Default command for GPU
CMD ["python", "-m", "carbon_aware_trainer.cli", "--serve", "--host", "0.0.0.0", "--port", "8080", "--gpu"]