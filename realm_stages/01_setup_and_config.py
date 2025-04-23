# realm_stages/01_setup_and_config.py
import os
import torch
import logging
import argparse
import yaml # Assuming load_config might need it, or directly uses it.

# Import existing components from the repository
from config.config_loader import load_config

# Setup logging (consistent with the original script)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- together.ai Platform Setup (Placeholder - Requires Implementation) ---
def setup_together_ai_environment(config: dict):
    """
    Configure the environment for running on the together.ai platform.
    **Action Required:** Fill this with together.ai specific setup based on config.
    """
    logger.info("Configuring environment for together.ai platform...")

    # 1. API Keys (Load from environment variables or secure storage)
    # Example: Check if keys are set
    together_api_key = os.getenv("TOGETHER_API_KEY")
    nim_api_key = os.getenv("NVIDIA_NIM_API_KEY") # Needed for NIMRewardModel

    if not together_api_key:
        logger.warning("TOGETHER_API_KEY environment variable not set.")
    if not nim_api_key:
        logger.warning("NVIDIA_NIM_API_KEY environment variable not set.")

    # 2. Resource Selection (Usually done via platform UI/config)
    logger.info("Ensure appropriate GPU resources are allocated via the together.ai platform.")

    # 3. Storage Configuration (Define paths based on config or platform)
    # Use paths from the loaded config where possible
    hf_home = config.get('environment', {}).get('hf_home', '/workspace/cache/huggingface')
    transformers_cache = config.get('environment', {}).get('transformers_cache', '/workspace/cache/transformers')
    output_base = config.get('output', {}).get('base_dir', '/workspace/outputs')

    os.environ["HF_HOME"] = os.getenv("HF_HOME", hf_home)
    os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", transformers_cache)
    logger.info(f"HF_HOME set to: {os.environ['HF_HOME']}")
    logger.info(f"TRANSFORMERS_CACHE set to: {os.environ['TRANSFORMERS_CACHE']}")
    logger.info(f"Base output directory configured to: {output_base}")
    os.makedirs(output_base, exist_ok=True) # Ensure base output dir exists

    # 4. Networking/Firewall
    logger.info("Verify network connectivity if external APIs (like NVIDIA NIM) are used.")

    # 5. Set other environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logger.info("together.ai environment setup placeholder complete. **Review and update required.**")


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Setup and Configuration for REALM Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the main configuration file.")
    args = parser.parse_args()

    # 1. Load Configuration
    logger.info(f"Loading configuration from: {args.config}")
    try:
        config = load_config(args.config)
        logger.info("Configuration loaded successfully.")
        # Optionally print parts of the config for verification
        # logger.info(f"Output directory: {config.get('output', {}).get('base_dir', 'N/A')}")
    except Exception as e:
        logger.error(f"Failed to load configuration from {args.config}: {e}", exc_info=True)
        return # Exit if config loading fails

    # 2. Setup Environment (together.ai specific + general)
    try:
        setup_together_ai_environment(config)
    except Exception as e:
        logger.error(f"Error during environment setup: {e}", exc_info=True)
        # Decide if failure here is critical
        # return

    # 3. Set Device
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        if not torch.cuda.is_available():
             logger.warning("CUDA not available, running on CPU. This might be slow.")
        # You might want to pass the device or config path to the next script,
        # but for now, each script will load the config and determine the device.
    except Exception as e:
        logger.error(f"Error setting torch device: {e}", exc_info=True)
        return

    logger.info("--- Stage 1: Setup and Configuration Complete ---")


if __name__ == "__main__":
    main()
