FROM python:3.11-slim

LABEL maintainer="Group 7 MLOps"
LABEL description="Telco Customer Churn Prediction API"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (leverages Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py train.py generate_synthetic_data.py ./

# Copy model artifacts (must run train.py locally first, or mount at runtime)
COPY models/ ./models/

# Data directory for optional retrain
COPY data/ ./data/

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
