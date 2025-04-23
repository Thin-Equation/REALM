#!/usr/bin/env python
# nim_ppo_finetune.py - Fine-tune LLM using only NIM reward model
import os
import logging
import yaml
import torch
import argparse
from typing import Dict, Any
from dotenv import load_dotenv

from trl import PPOTrainer, PPOConfig, create_reference_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

from models.nim_reward import NIMRewardModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

class NIMPPOTrainer:
    """
    PPO Trainer that uses NIM reward model directly for LLM fine-tuning
    """
    
    def __init__(
        self,
        config: Dict,
        nim_reward_model: NIMRewardModel,
        device = None
    ):
        self.config = config
        self.nim_reward_model = nim_reward_model
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model
        model_name = config["rlhf"]["ppo"]["model_name"]
        logger.info(f"Loading model and tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure PPO
        ppo_config = PPOConfig(
            model_name=config["rlhf"]["ppo"]["model_name"],
            learning_rate=config["rlhf"]["ppo"]["learning_rate"],
            batch_size=config["rlhf"]["ppo"]["batch_size"],
            mini_batch_size=config["rlhf"]["ppo"]["mini_batch_size"],
            gradient_accumulation_steps=config["rlhf"]["ppo"]["gradient_accumulation_steps"],
            optimize_cuda_cache=True,
            early_stopping=True,
            target_kl=0.1,
            kl_penalty=config["rlhf"]["ppo"]["kl_penalty"],
            seed=config["training"]["seed"]
        )
        
        # Create a reference model for KL divergence
        ref_model = create_reference_model(self.model)
        
        # Initialize the PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=ref_model,
            tokenizer=self.tokenizer,
            dataset=None,  # We'll use a custom data iteration
            data_collator=None
        )
        
        logger.info("NIM PPO Trainer initialized")
    
    def _get_reward(self, prompt: str, response: str) -> float:
        """Get reward directly from NIM reward model"""
        return self.nim_reward_model.get_reward_score(prompt, response)
    
    def _generate_responses(self, prompts: list, max_length: int = 512) -> list:
        """Generate responses from the model"""
        responses = []
        
        for prompt in tqdm(prompts, desc="Generating responses"):
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate with current policy model
            outputs = self.model.generate(
                inputs, 
                max_length=max_length, 
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):]  # Remove the prompt
            
            responses.append(response)
        
        return responses
    
    def train(self, dataset_path: str = None, max_steps: int = 1000) -> None:
        """
        Train the model using PPO with NIM reward model
        
        Args:
            dataset_path: Path to dataset file (JSON with 'prompt' key)
            max_steps: Maximum number of training steps
        """
        # Load dataset
        if dataset_path:
            import json
            with open(dataset_path, "r") as f:
                dataset = json.load(f)
            prompts = dataset["prompt"]
        else:
            # Use SHP dataset as fallback
            logger.info("No dataset provided, using SHP dataset")
            dataset = load_dataset("stanfordnlp/SHP")
            prompts = [item["post"] for item in dataset["train"]]
        
        logger.info(f"Loaded {len(prompts)} prompts for training")
        
        # Training loop
        step = 0
        batch_size = self.config["rlhf"]["ppo"]["batch_size"]
        
        for i in tqdm(range(0, len(prompts), batch_size)):
            # Check if max_steps is reached
            if step >= max_steps:
                logger.info(f"Reached maximum steps {max_steps}")
                break
            
            # Get batch of prompts
            batch_prompts = prompts[i:i+batch_size]
            
            # Generate responses
            batch_responses = self._generate_responses(batch_prompts)
            
            # Get rewards directly from NIM model
            batch_rewards = [self._get_reward(p, r) for p, r in zip(batch_prompts, batch_responses)]
            
            # Prepare inputs for PPO step
            query_tensors = [
                self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                for prompt in batch_prompts
            ]
            
            response_tensors = [
                self.tokenizer.encode(response, return_tensors="pt").to(self.device)
                for response in batch_responses
            ]
            
            # Run PPO step
            stats = self.ppo_trainer.step(query_tensors, response_tensors, batch_rewards)
            
            # Log stats
            logger.info(f"Step {step+1}: {stats}")
            step += 1
        
        logger.info("PPO training with NIM rewards completed")
    
    def save_model(self, output_dir: str) -> None:
        """Save the fine-tuned model and tokenizer"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM using NIM reward model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to dataset for RLHF")
    parser.add_argument("--output_dir", type=str, default="models/nim_ppo_finetuned", help="Directory to save the fine-tuned model")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of training steps")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize NIM reward model
    nim_reward_model = NIMRewardModel(
        api_key=config["nim_reward"]["api_key"],
        base_url=config["nim_reward"]["base_url"],
        model_id=config["nim_reward"]["model_id"],
        max_retries=config["nim_reward"]["max_retries"],
        retry_delay=config["nim_reward"]["retry_delay"]
    )
    
    # Initialize NIM PPO trainer
    trainer = NIMPPOTrainer(
        config=config,
        nim_reward_model=nim_reward_model
    )
    
    # Train model
    trainer.train(
        dataset_path=args.dataset_path,
        max_steps=args.max_steps
    )
    
    # Save model
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()