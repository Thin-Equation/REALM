# Dockerfile for Together AI with GPU support
# Use a PyTorch base image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory to match Together AI conventions
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages for GPU acceleration
RUN pip install --no-cache-dir \
    ninja \
    bitsandbytes \
    flash-attn \
    triton

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/models
RUN mkdir -p /workspace/logs
RUN mkdir -p /workspace/cache
RUN mkdir -p /workspace/outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_NIM_API_KEY=${NVIDIA_NIM_API_KEY}
ENV GEMINI_API_KEY=${GEMINI_API_KEY}
ENV MODEL_PATH=/workspace/models/final_model.pt

# Set environment variables for PyTorch to use GPU
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV CUDA_VISIBLE_DEVICES=0

# Expose port for API (if needed)
EXPOSE 8000

# The default command will be overridden by Together AI job submission
# But we provide a sensible default for local testing
CMD ["python", "api_server.py"]
