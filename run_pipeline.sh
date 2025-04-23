#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the root directory (assuming the script is run from the project root)
ROOT_DIR=$(pwd)
STAGES_DIR="${ROOT_DIR}/realm_stages"
CONFIG_FILE="${ROOT_DIR}/config/config.yaml" # Default config path

echo "==========================================="
echo "Starting REALM Pipeline Execution"
echo "Using Config: ${CONFIG_FILE}"
echo "==========================================="

# Stage 1: Setup and Configuration Check
echo ""
echo "--- Stage 1: Setup and Configuration ---"
python "${STAGES_DIR}/01_setup_and_config.py" --config "${CONFIG_FILE}"
echo "--- Stage 1 Complete ---"

# Stage 2: Train REALM Linear Model
echo ""
echo "--- Stage 2: Train REALM Linear Model ---"
python "${STAGES_DIR}/02_train_realm_linear_model.py" --config "${CONFIG_FILE}"
echo "--- Stage 2 Complete ---"

# Stage 3: Run PPO with REALM Reward
echo ""
echo "--- Stage 3: Run PPO with REALM Reward ---"
python "${STAGES_DIR}/03_run_ppo_realm.py" --config "${CONFIG_FILE}"
echo "--- Stage 3 Complete ---"

# Stage 4: Run PPO with Standard Reward
echo ""
echo "--- Stage 4: Run PPO with Standard Reward ---"
python "${STAGES_DIR}/04_run_ppo_standard.py" --config "${CONFIG_FILE}"
echo "--- Stage 4 Complete ---"

# Stage 5: Evaluate Models
echo ""
echo "--- Stage 5: Evaluate Models ---"
python "${STAGES_DIR}/05_evaluate_models.py" --config "${CONFIG_FILE}"
echo "--- Stage 5 Complete ---"

echo ""
echo "==========================================="
echo "REALM Pipeline Execution Finished"
echo "==========================================="

