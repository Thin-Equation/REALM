#!/usr/bin/env python
# verify_models.py - Check that all required model weights are saved
import os
import logging
import argparse
import torch
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_model_file(model_path, model_name):
    """Check if a model file exists and has the correct format"""
    if not os.path.exists(model_path):
        logger.error(f"❌ {model_name} file not found at {model_path}")
        return False
    
    try:
        if model_path.endswith('.pt'):
            # Try to load the PyTorch model
            _ = torch.load(model_path, map_location='cpu')
            logger.info(f"✓ {model_name} file exists and is valid at {model_path}")
        else:
            # Check if directory contains expected files for transformers models
            config_file = os.path.join(model_path, "config.json")
            model_file = os.path.join(model_path, "pytorch_model.bin")
            
            if not os.path.exists(config_file):
                logger.warning(f"⚠️ {model_name} may be incomplete: config.json not found in {model_path}")
                return False
                
            if not os.path.exists(model_file) and not any(f.endswith('.safetensors') for f in os.listdir(model_path)):
                logger.warning(f"⚠️ {model_name} may be incomplete: neither pytorch_model.bin nor safetensors found in {model_path}")
                return False
                
            logger.info(f"✓ {model_name} directory exists and contains required files at {model_path}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to validate {model_name} at {model_path}: {e}")
        return False

def get_model_size(path):
    """Get the size of a model file or directory in MB"""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    elif os.path.isdir(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)
    return 0

def main():
    parser = argparse.ArgumentParser(description="Verify that all model weights are saved")
    parser.add_argument("--reward_model_path", type=str, default="models/final_model.pt", 
                        help="Path to the trained reward model")
    parser.add_argument("--combined_model_path", type=str, default="models/ppo_finetuned", 
                        help="Path to the model fine-tuned with combined reward")
    parser.add_argument("--nim_model_path", type=str, default="models/nim_ppo_finetuned", 
                        help="Path to the model fine-tuned with NIM reward")
    args = parser.parse_args()
    
    logger.info("Verifying all model weights are saved correctly...")
    
    # Check reward model
    reward_model_ok = check_model_file(args.reward_model_path, "Reward model (LinearRewardModel)")
    if reward_model_ok:
        size_mb = get_model_size(args.reward_model_path)
        logger.info(f"  Reward model size: {size_mb:.2f} MB")
    
    # Check combined-reward fine-tuned model
    combined_model_ok = check_model_file(args.combined_model_path, "Combined-reward fine-tuned model")
    if combined_model_ok:
        size_mb = get_model_size(args.combined_model_path)
        logger.info(f"  Combined-reward fine-tuned model size: {size_mb:.2f} MB")
    
    # Check NIM-reward fine-tuned model
    nim_model_ok = check_model_file(args.nim_model_path, "NIM-reward fine-tuned model")
    if nim_model_ok:
        size_mb = get_model_size(args.nim_model_path)
        logger.info(f"  NIM-reward fine-tuned model size: {size_mb:.2f} MB")
    
    # Overall status
    all_models_ok = reward_model_ok and combined_model_ok and nim_model_ok
    
    if all_models_ok:
        logger.info("✅ All three required model weights are saved correctly!")
        logger.info("  1. Reward Model (LinearRewardModel): " + args.reward_model_path)
        logger.info("  2. Combined-Reward Fine-tuned LLM: " + args.combined_model_path)
        logger.info("  3. NIM-Reward Fine-tuned LLM: " + args.nim_model_path)
        
        # Remind about downloading models from Brev
        logger.info("\nTo download these models from your Brev instance, run:")
        logger.info(f"  brev pull instance-name:/app/{args.reward_model_path} ./local-path/")
        logger.info(f"  brev pull -r instance-name:/app/{args.combined_model_path} ./local-path/combined/")
        logger.info(f"  brev pull -r instance-name:/app/{args.nim_model_path} ./local-path/nim/")
    else:
        logger.error("❌ Some required model weights are missing!")
        if not reward_model_ok:
            logger.error("  - Reward model is missing. Run: python main.py --mode train")
        if not combined_model_ok:
            logger.error("  - Combined-reward fine-tuned model is missing. Run: python main.py --mode ppo --model_path models/final_model.pt")
        if not nim_model_ok:
            logger.error("  - NIM-reward fine-tuned model is missing. Run: python nim_ppo_finetune.py")

if __name__ == "__main__":
    main()