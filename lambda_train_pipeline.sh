#!/bin/bash
# lambda_train_pipeline.sh - Complete training pipeline for REALM on Lambda Labs

# Exit on error
set -e

# Setup logging
LOG_FILE="lambda_training.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "$(date): Starting REALM training pipeline on Lambda Labs"

# Check for NVIDIA_NIM_API_KEY
if [ -z "$NVIDIA_NIM_API_KEY" ]; then
    echo "ERROR: NVIDIA_NIM_API_KEY environment variable is not set"
    echo "Please set it using: export NVIDIA_NIM_API_KEY=your_api_key"
    exit 1
fi

# Check GPU availability and resources
if [ -x "$(command -v nvidia-smi)" ]; then
    echo "GPU information:"
    nvidia-smi
    echo "GPU Memory Usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv
else
    echo "ERROR: NVIDIA drivers not detected. Lambda Labs instances should have GPU support."
    echo "Please contact Lambda Labs support if you're seeing this error."
    exit 1
fi

# Display Lambda Labs instance information
echo "System Information:"
cat /etc/os-release
echo "CPU Info:"
lscpu | grep "Model name"
echo "Total Memory:"
free -h | grep "Mem:"

# Create directories
mkdir -p models
mkdir -p models/checkpoints
mkdir -p evaluation_results

# Setup Python environment - Lambda Labs sometimes uses conda environments
if [ -x "$(command -v conda)" ]; then
    echo "Conda detected, ensuring dependencies are installed..."
    # Optional: Activate specific conda environment if needed
    # conda activate your_env_name
    pip install -r requirements.txt
else
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Configure Weights & Biases for experiment tracking (optional)
# If you're using W&B with Lambda Labs
if [ ! -z "$WANDB_API_KEY" ]; then
    echo "Configuring Weights & Biases with provided API key..."
    wandb login $WANDB_API_KEY
else
    echo "WANDB_API_KEY not set, running in offline mode or without W&B tracking"
    # Uncomment to force offline mode
    # export WANDB_MODE=offline
fi

# Display training configuration
echo "Training with the following configuration:"
cat config/config.yaml

echo "$(date): Step 1/4 - Training the Linear Reward Model..."
python main.py --mode train --config config/config.yaml
echo "$(date): Linear Reward Model training completed"

echo "$(date): Step 2/4 - Fine-tuning with custom reward model using PPO..."
python main.py --mode ppo --model_path models/final_model.pt
echo "$(date): Custom reward model fine-tuning completed"

echo "$(date): Step 3/4 - Fine-tuning with NIM reward model using PPO..."
python main.py --mode nim_ppo --config config/config.yaml --output_dir models/nim_ppo_finetuned --max_steps 1000
echo "$(date): NIM reward model fine-tuning completed"

echo "$(date): Step 4/4 - Evaluating both models on TruthfulQA dataset..."
python main.py --mode evaluate --combined_model_path models/ppo_finetuned --nim_model_path models/nim_ppo_finetuned --output_dir evaluation_results
echo "$(date): Evaluation completed"

echo "$(date): Verifying all model weights were saved correctly..."
python main.py --mode verify

echo "$(date): All training steps completed. Results summary:"

# Display evaluation results summary if available
if [ -f "evaluation_results/metrics.json" ]; then
    echo "Evaluation metrics:"
    cat evaluation_results/metrics.json
else
    echo "Evaluation metrics file not found"
fi

# Compress results for easier download from Lambda Labs
echo "$(date): Compressing model weights and results for download..."
tar -czvf realm_results.tar.gz models/ evaluation_results/ lambda_training.log

echo ""
echo "Training pipeline completed. Your results are compressed in realm_results.tar.gz"
echo ""
echo "To download results from Lambda Labs:"
echo "  1. Use the Lambda Labs web console to download realm_results.tar.gz, or"
echo "  2. Use SCP: scp <username>@<lambda-instance-ip>:/path/to/realm_results.tar.gz /local/path/"
echo ""
echo "$(date): Pipeline execution completed"

# Optional: Shutdown instance when done (if you want to save credits)
# echo "Shutting down instance in 5 minutes to save credits. Press Ctrl+C to cancel."
# sleep 300 && sudo poweroff