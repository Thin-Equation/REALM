#!/bin/bash
# brev_train_pipeline.sh - Complete training pipeline for REALM on Brev

# Exit on error
set -e

# Setup logging
LOG_FILE="brev_training.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "$(date): Starting REALM training pipeline on Brev"

# Check for NVIDIA_NIM_API_KEY
if [ -z "$NVIDIA_NIM_API_KEY" ]; then
    echo "ERROR: NVIDIA_NIM_API_KEY environment variable is not set"
    echo "Please set it using: export NVIDIA_NIM_API_KEY=your_api_key"
    exit 1
fi

# Check GPU availability
if [ -x "$(command -v nvidia-smi)" ]; then
    echo "GPU information:"
    nvidia-smi
else
    echo "WARNING: NVIDIA drivers not detected. Training may be slow without GPU acceleration."
fi

# Create directories
mkdir -p models
mkdir -p models/checkpoints
mkdir -p evaluation_results

echo "$(date): Step 1/4 - Training the Linear Reward Model..."
python main.py --mode train --config config/config.yaml
echo "$(date): Linear Reward Model training completed"

echo "$(date): Step 2/4 - Fine-tuning with custom reward model using PPO..."
python main.py --mode ppo --model_path models/final_model.pt
echo "$(date): Custom reward model fine-tuning completed"

echo "$(date): Step 3/4 - Fine-tuning with NIM reward model using PPO..."
python nim_ppo_finetune.py --config config/config.yaml --output_dir models/nim_ppo_finetuned --max_steps 1000
echo "$(date): NIM reward model fine-tuning completed"

echo "$(date): Step 4/4 - Evaluating both models on TruthfulQA dataset..."
python evaluate_models.py --combined_model_path models/ppo_finetuned --nim_model_path models/nim_ppo_finetuned --output_dir evaluation_results
echo "$(date): Evaluation completed"

echo "$(date): Verifying all model weights were saved correctly..."
python verify_models.py

echo "$(date): All training steps completed. Results summary:"

# Display evaluation results summary if available
if [ -f "evaluation_results/metrics.json" ]; then
    echo "Evaluation metrics:"
    cat evaluation_results/metrics.json
else
    echo "Evaluation metrics file not found"
fi

echo ""
echo "To download all model weights and evaluation results from Brev, run:"
echo "  brev pull <instance-name>:/app/models/final_model.pt ./models/"
echo "  brev pull -r <instance-name>:/app/models/ppo_finetuned ./models/"
echo "  brev pull -r <instance-name>:/app/models/nim_ppo_finetuned ./models/"
echo "  brev pull -r <instance-name>:/app/evaluation_results ./evaluation_results/"
echo ""
echo "Replace <instance-name> with your Brev instance name"
echo ""
echo "$(date): Pipeline execution completed"