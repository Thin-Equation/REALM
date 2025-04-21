# main.
# 

import os
import argparse
import logging
import yaml
import torch
import random
import numpy as np
from typing import Dict, Any
from dotenv import load_dotenv

from utils.validation import validate_environment
from utils.embedding_utils import LajavanessEmbedding
from data.processors import SHPDataProcessor, create_dataloaders
from models.reward_model import LinearRewardModel
from training.trainer import RewardModelTrainer
from inference.predictor import RewardPredictor
from rlhf.ppo_integration import PPOTrainerWithCustomReward
from rlhf.dpo_integration import DPOTrainerWithCustomReward
from models.nim_reward import NIMRewardModel

# Load environment variables
load_dotenv()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Replace environment variables in config
    for section in config:
        if isinstance(config[section], dict):
            for key, value in config[section].items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    config[section][key] = os.environ.get(env_var, "")
    
    return config

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Combined Reward Model for RLHF")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "ppo", "dpo", "predict"], default="train", help="Mode to run")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint (required for eval, ppo, dpo, predict)")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for prediction (used in predict mode)")
    parser.add_argument("--response", type=str, default=None, help="Response for prediction (used in predict mode)")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to dataset for RLHF (used in ppo/dpo mode)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # Validate environment first
    logger.info("Validating environment...")
    validate_environment()
    
    # Setup logging
    logger.info("Starting Combined Reward Model")
    logger.info(f"Running in {args.mode} mode")
    
    # Set random seed
    set_seed(config["training"]["seed"])
    
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize Llama 3.1 Nemotron Reward model via NIM API
    nim_reward_model = NIMRewardModel(
        api_key=config["nim_reward"]["api_key"],
        base_url=config["nim_reward"]["base_url"],
        model_id=config["nim_reward"]["model_id"],
        max_retries=config["nim_reward"]["max_retries"],
        retry_delay=config["nim_reward"]["retry_delay"]
    )
    
    # Initialize Lajavaness Embedding
    embedding_model = LajavanessEmbedding(
        model_id=config["embedding"]["model_id"]
    )
    
    if args.mode == "train":
        # Load and process dataset
        data_processor = SHPDataProcessor(config)
        train_data, val_data, test_data = data_processor.load_dataset()
        
        # Create dataloaders
        train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
            config, train_data, val_data, test_data, nim_reward_model, embedding_model
        )
        
        # Initialize model
        model = LinearRewardModel(
            input_dim=config["model"]["input_dim"],
            hidden_dims=config["model"]["hidden_dims"],
            output_dim=config["model"]["output_dim"],
            dropout=config["model"]["dropout"]
        )
        
        # Initialize trainer
        trainer = RewardModelTrainer(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            device=device
        )
        
        # Train model
        trained_model = trainer.train()
        
        # Save final model
        final_model_path = os.path.join("models", "final_model.pt")
        trained_model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
    elif args.mode == "eval":
        # Check if model path is provided
        if args.model_path is None:
            raise ValueError("Model path must be provided for eval mode")
        
        # Load and process dataset
        data_processor = SHPDataProcessor(config)
        _, _, test_data = data_processor.load_dataset()
        
        # Create test dataloader
        _, _, test_dataloader = create_dataloaders(
            config, None, None, test_data, nim_reward_model, embedding_model
        )
        
        # Load model
        model = LinearRewardModel.load(args.model_path, device=device)
        
        # Initialize trainer
        trainer = RewardModelTrainer(
            model=model,
            config=config,
            train_dataloader=None,
            val_dataloader=None,
            test_dataloader=test_dataloader,
            device=device
        )
        
        # Evaluate model
        test_metrics = trainer.test()
        logger.info(f"Test results: {test_metrics}")
        
    elif args.mode == "predict":
        # Check if model path, prompt, and response are provided
        if args.model_path is None:
            raise ValueError("Model path must be provided for predict mode")
        if args.prompt is None or args.response is None:
            raise ValueError("Prompt and response must be provided for predict mode")
        
        # Initialize predictor
        predictor = RewardPredictor(
            model_path=args.model_path,
            nim_reward_model=nim_reward_model,
            embedding_model=embedding_model,
            device=device
        )
        
        # Predict reward
        reward = predictor.predict(args.prompt, args.response)
        logger.info(f"Predicted reward: {reward}")
        
    elif args.mode == "ppo":
        # Check if model path is provided
        if args.model_path is None:
            raise ValueError("Model path must be provided for PPO mode")
        
        # Initialize predictor
        predictor = RewardPredictor(
            model_path=args.model_path,
            nim_reward_model=nim_reward_model,
            embedding_model=embedding_model,
            device=device
        )
        
        # Initialize PPO trainer
        ppo_trainer = PPOTrainerWithCustomReward(
            config=config,
            reward_predictor=predictor,
            device=device
        )
        
        # Load dataset
        if args.dataset_path:
            import json
            with open(args.dataset_path, "r") as f:
                dataset = json.load(f)
        else:
            # Use SHP dataset for demonstration
            data_processor = SHPDataProcessor(config)
            train_data, _, _ = data_processor.load_dataset()
            dataset = {"prompt": [item["post"] for item in train_data]}
        
        # Train with PPO
        ppo_trainer.train(
            dataset=dataset,
            num_epochs=1,
            max_steps=100  # Limit steps for demonstration
        )
        
        # Save the fine-tuned model
        ppo_trainer.save_model(os.path.join("models", "ppo_finetuned"))
        
    elif args.mode == "dpo":
        # Check if model path is provided
        if args.model_path is None:
            raise ValueError("Model path must be provided for DPO mode")
        
        # Initialize predictor
        predictor = RewardPredictor(
            model_path=args.model_path,
            nim_reward_model=nim_reward_model,
            embedding_model=embedding_model,
            device=device
        )
        
        # Initialize DPO trainer
        dpo_trainer = DPOTrainerWithCustomReward(
            config=config,
            reward_predictor=predictor,
            device=device
        )
        
        # Load dataset
        if args.dataset_path:
            import json
            with open(args.dataset_path, "r") as f:
                dataset = json.load(f)
            
            dpo_trainer.train(
                dataset=dataset,
                num_epochs=3,
                generate_pairs=True
            )
        else:
            # Use SHP dataset for demonstration
            data_processor = SHPDataProcessor(config)
            train_data, _, _ = data_processor.load_dataset()
            
            # Convert to format expected by DPO trainer
            dataset = {"prompt": [item["post"] for item in train_data]}
            
            dpo_trainer.train(
                dataset=dataset,
                num_epochs=3,
                generate_pairs=True
            )

if __name__ == "__main__":
    main()
