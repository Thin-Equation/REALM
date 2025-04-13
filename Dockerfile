# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Create directories
RUN mkdir -p /app/models/checkpoints
RUN mkdir -p /app/logs
RUN mkdir -p /app/.cache/huggingface

# Default command
CMD ["python", "main.py", "--config", "config/config.yaml", "--mode", "train"]
