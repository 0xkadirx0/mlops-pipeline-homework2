# MLOps CI/CD Pipeline - Dockerfile
# This Dockerfile packages the ML model and serving code into a deployable artifact

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY mlops_pipeline.py .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from mlops_pipeline import SmokeTest; SmokeTest.test_service_health()" || exit 1

# Default command: Run tests
CMD ["python", "mlops_pipeline.py"]

# Labels for metadata
LABEL maintainer="MLOps Student"
LABEL description="MLOps CI/CD Pipeline - High-Cardinality Prediction Service"
LABEL version="1.0.0"
