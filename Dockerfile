# Dockerfile for REALM Pipeline on Together AI
# Base image with CUDA support - PyTorch 2.0.1 with CUDA 11.7
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory to match Together AI conventions
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    htop \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install GPU acceleration packages
RUN pip install --no-cache-dir \
    ninja \
    bitsandbytes \
    flash-attn \
    triton

# Setup directory structure optimized for pipeline stages
RUN mkdir -p /workspace/models/checkpoints \
    && mkdir -p /workspace/outputs/metrics \
    && mkdir -p /workspace/logs \
    && mkdir -p /workspace/cache/huggingface \
    && mkdir -p /workspace/cache/transformers \
    && mkdir -p /workspace/cache/datasets

# Set environment variables for HuggingFace and Transformers
ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/workspace/cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/cache/transformers \
    DATASETS_CACHE=/workspace/cache/datasets \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" \
    NVIDIA_VISIBLE_DEVICES=0

# Copy configuration files first
COPY config/ /workspace/config/

# Copy utility code and models
COPY utils/ /workspace/utils/
COPY models/ /workspace/models/
COPY data/ /workspace/data/

# Copy pipeline stage scripts
COPY realm_stages/ /workspace/realm_stages/

# Copy remaining application code
COPY *.py /workspace/

# Make scripts executable
RUN chmod +x /workspace/realm_stages/*.py

# Set additional environment variables for API keys
ENV NVIDIA_NIM_API_KEY=${NVIDIA_NIM_API_KEY} \
    GEMINI_API_KEY=${GEMINI_API_KEY} \
    MODEL_PATH=/workspace/models/final_model.pt

# Expose port for API (if needed)
EXPOSE 8000

# Default command - this will be overridden by Together AI job submissions
# Each stage script will be run individually via the Together AI platform
CMD ["python", "/workspace/realm_stages/01_setup_and_config.py", "--config", "config/config.yaml"]
