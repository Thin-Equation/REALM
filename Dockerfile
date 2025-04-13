# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models
RUN mkdir -p /app/logs
RUN mkdir -p /app/cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_NIM_API_KEY=${NVIDIA_NIM_API_KEY}
ENV GEMINI_API_KEY=${GEMINI_API_KEY}
ENV MODEL_PATH=/app/models/final_model.pt

# Expose port for API
EXPOSE 8000

# Set command
CMD ["python", "api_server.py"]
