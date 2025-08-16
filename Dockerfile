
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml .
COPY src/ ./src/
COPY README.md .

# Install package
RUN pip install --no-cache-dir -e .

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 carbonuser && chown -R carbonuser:carbonuser /app
USER carbonuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import carbon_aware_trainer; print('healthy')" || exit 1

# Default command
CMD ["python", "-m", "carbon_aware_trainer.cli"]
