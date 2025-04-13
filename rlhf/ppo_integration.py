# rlhf/ppo_integration.py
import os
import torch
import logging
from typing import Dict, List, Callable, Optional, Union
from trl import PPOTrainer, PPOConfig, create_reference_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from models.reward_model import LinearRewardModel
from models.llama_reward import LlamaRewardModel
from utils.embedding_utils import GeminiEmbedding
from inference.predictor import RewardPredictor

logger = logging.getLogger(__name__)

class PPOTrainerWithCustomReward:
    """
    PPO Trainer with custom combined reward model for LLM fine-tuning
    """
    
    def __init__(
        self,
        config: Dict,
        reward_predictor: RewardPredictor,
        tokenizer=None,
        model=None,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.reward_predictor = reward_predictor
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model if not provided
        if tokenizer is None or model is None:
            model_name = config["rlhf"]["ppo"]["model_name"]
            logger.info(f"Loading model and tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer
            self.model = model
        
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
        
        logger.info("PPO Trainer with custom reward initialized")
    
    def _get_reward(self, prompt: str, response: str) -> float:
        """Get reward from the custom reward predictor"""
        return self.reward_predictor.predict(prompt, response)
    
    def _generate_responses(self, prompts: List[str], max_length: int = 512) -> List[str]:
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
    
    def train(
        self, 
        dataset: Dict[str, List[str]], 
        num_epochs: int = 1,
        max_steps: Optional[int] = None
    ) -> None:
        """
        Train the model using PPO with the custom reward model
        
        Args:
            dataset: Dictionary with 'prompt' key containing prompts
            num_epochs: Number of epochs to train
            max_steps: Maximum number of training steps (overrides epochs if provided)
        """
        prompts = dataset["prompt"]
        step = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            for i in tqdm(range(0, len(prompts), self.config["rlhf"]["ppo"]["batch_size"])):
                # Check if max_steps is reached
                if max_steps is not None and step >= max_steps:
                    logger.info(f"Reached maximum steps {max_steps}")
                    return
                
                # Get batch of prompts
                batch_prompts = prompts[i:i+self.config["rlhf"]["ppo"]["batch_size"]]
                
                # Generate responses
                batch_responses = self._generate_responses(batch_prompts)
                
                # Get rewards
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
        
        logger.info("PPO training completed")
    
    def save_model(self, output_dir: str) -> None:
        """Save the fine-tuned model and tokenizer"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to {output_dir}")
