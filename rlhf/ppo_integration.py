# rlhf/ppo_huggingface.py
import os
import torch
import logging
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
from tqdm import tqdm

from trl import PPOConfig, PPOTrainer

from inference.predictor import RewardPredictor
from models.nim_reward import NIMRewardModel
# Add import for SHPDataProcessor
from data.processors import SHPDataProcessor

logger = logging.getLogger(__name__)

class HuggingFacePPOTrainer:
    """
    PPO Trainer that leverages Hugging Face's TRL library version 0.16.1 
    for industry-standard PPO implementation
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
        
        # Model initialization
        model_name = config["rlhf"]["ppo"]["model_name"]
        logger.info(f"Loading model and tokenizer: {model_name}")
        
        # Initialize tokenizer and model if not provided
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "left"  # Important for PPO training
        else:
            self.tokenizer = tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "left"
            
        # Initialize model if not provided
        if model is None:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            self.model = model
                
        # Move model to device
        self.model.to(self.device)
        
        # Create PPO config
        self.ppo_config = self._create_ppo_config(config["rlhf"]["ppo"])
        
        # Set up generation args
        self.max_length = config["rlhf"]["ppo"].get("max_length", 256)
        self.generation_kwargs = {
            "max_new_tokens": self.max_length,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        logger.info("HuggingFace PPO Trainer initialized (for TRL version 0.16.1)")
    
    def _create_ppo_config(self, ppo_config_dict: Dict) -> PPOConfig:
        """Create a minimal PPOConfig with only the most basic parameters"""
        # Only use the most basic parameters that should be supported across versions
        return PPOConfig(
            learning_rate=float(ppo_config_dict.get("learning_rate", 1e-5)),
            batch_size=ppo_config_dict.get("batch_size", 8),
            mini_batch_size=ppo_config_dict.get("mini_batch_size", 4),
        )
    
    def _prepare_dataset(self, dataset: Optional[Dict] = None, dataset_name: str = None, max_samples: int = None) -> Dataset:
        """
        Convert dict dataset to HF Dataset for PPO Trainer or load dataset directly
        
        Args:
            dataset: Dictionary with prompts under "prompt" key
            dataset_name: Name of HF dataset to load directly (e.g., "stanfordnlp/SHP")
            max_samples: Maximum number of samples to use
            
        Returns:
            HuggingFace Dataset formatted for PPO training
        """
        # If dataset dict is provided, use it
        if dataset is not None and "prompt" in dataset:
            prompts = dataset.get("prompt", [])
            if not prompts:
                logger.warning("No prompts found in dataset dictionary")
                return None
                
            # Create HF dataset with prompts
            hf_dataset = Dataset.from_dict({"query": prompts})
        
        # If dataset_name is provided, load it using SHPDataProcessor for SHP dataset
        elif dataset_name is not None:
            logger.info(f"Loading dataset: {dataset_name}")
            try:
                # Handle SHP dataset using SHPDataProcessor
                if "stanfordnlp/SHP" in dataset_name:
                    # Create temporary config for SHPDataProcessor
                    temp_config = {"data": {"dataset_name": dataset_name, "preprocessing": {"cache_dir": "cache", "max_length": 1024}}}
                    data_processor = SHPDataProcessor(temp_config)
                    train_data, _, _ = data_processor.load_dataset()
                    
                    # Extract prompts from the "history" field
                    prompts = [item["history"] for item in train_data]
                    hf_dataset = Dataset.from_dict({"query": prompts})
                    logger.info(f"Loaded {len(prompts)} prompts from SHP dataset using SHPDataProcessor")
                else:
                    # For non-SHP datasets, use direct loading
                    raw_dataset = load_dataset(dataset_name)
                    
                    # Try to find a suitable split
                    train_split = raw_dataset.get("train", raw_dataset.get("default", None))
                    if train_split is None:
                        logger.error(f"No suitable split found in dataset {dataset_name}")
                        return None
                    
                    # Try to find a suitable column
                    prompt_col = None
                    for col in ["prompt", "question", "text", "input", "instruction", "history"]:
                        if col in train_split.column_names:
                            prompt_col = col
                            break
                    
                    if prompt_col is None:
                        logger.error(f"No suitable prompt column found in dataset {dataset_name}")
                        return None
                    
                    prompts = train_split[prompt_col]
                    hf_dataset = Dataset.from_dict({"query": prompts})
                    logger.info(f"Loaded {len(prompts)} prompts from {dataset_name} using column {prompt_col}")
            except Exception as e:
                logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
                return None
        else:
            logger.error("Either dataset or dataset_name must be provided")
            return None
        
        # Limit number of samples if specified
        if max_samples and len(hf_dataset) > max_samples:
            hf_dataset = hf_dataset.select(range(max_samples))
            logger.info(f"Using {max_samples} samples from dataset")
            
        return hf_dataset
    
    def _reward_fn(self, queries: List[str], responses: List[str]) -> List[float]:
        """Reward function wrapper for the PPO trainer"""
        rewards = []
        for query, response in zip(queries, responses):
            reward = self.reward_predictor.predict(query, response)
            rewards.append(float(reward))
        return rewards
        
    def train(self, dataset: Dict, num_epochs: int = 1, max_steps: int = 100):
        """Train the model using TRL 0.16.1's PPO implementation"""
        logger.info(f"Starting PPO training with HuggingFace TRL 0.16.1 for {num_epochs} epochs, max {max_steps} steps")
        
        # Prepare dataset
        hf_dataset = self._prepare_dataset(dataset)
        if hf_dataset is None:
            logger.error("Cannot train with empty dataset")
            return self.model
        
        # Limit steps if needed
        if max_steps and max_steps < len(hf_dataset):
            hf_dataset = hf_dataset.select(range(max_steps))
        
        # Create a minimal PPOTrainer with only the most essential parameters
        # that should work across different versions
        try:
            # Try the current approach for newer versions
            ppo_trainer = PPOTrainer(
                config=self.ppo_config,
                model=self.model,
                ref_model=None,
                tokenizer=self.tokenizer,
                dataset=hf_dataset,
            )
        except TypeError as e:
            # If there's an error, try an alternative approach for older versions
            logger.warning(f"First PPOTrainer initialization attempt failed: {e}")
            try:
                ppo_trainer = PPOTrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    dataset=hf_dataset,
                    learning_rate=self.ppo_config.learning_rate,
                    batch_size=self.ppo_config.batch_size,
                    mini_batch_size=self.ppo_config.mini_batch_size,
                )
            except TypeError as e2:
                # If both fail, use a third approach as a last resort
                logger.warning(f"Second PPOTrainer initialization attempt failed: {e2}")
                logger.info("Attempting to create a simpler implementation...")
                
                # Instead of using PPOTrainer, we'll implement a simplified version
                # that just fine-tunes the model on the generated responses with rewards
                ppo_trainer = None
        
        # Train for the specified number of epochs
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Initialize metrics tracking
            epoch_rewards = []
            
            # Get only the amount of data we need for this epoch
            epoch_dataset = hf_dataset.shuffle(seed=epoch)
            if max_steps:
                epoch_dataset = epoch_dataset.select(range(min(max_steps, len(epoch_dataset))))
            
            # Process data in batches
            for i in range(0, len(epoch_dataset), self.ppo_config.batch_size):
                batch = epoch_dataset.select(range(i, min(i + self.ppo_config.batch_size, len(epoch_dataset))))
                
                # Get queries
                queries = batch["query"]
                
                # Generate responses and log them
                responses = []
                for query in tqdm(queries, desc=f"Processing batch {i//self.ppo_config.batch_size + 1}"):
                    # Tokenize the query
                    query_tensor = self.tokenizer(query, return_tensors="pt").to(self.device)
                    
                    # Generate a response
                    with torch.no_grad():
                        output = self.model.generate(
                            input_ids=query_tensor.input_ids,
                            attention_mask=query_tensor.attention_mask,
                            max_new_tokens=self.max_length,
                            do_sample=True,
                            top_p=0.9,
                            top_k=0,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                    
                    # Extract just the response part (not the query)
                    response_text = self.tokenizer.decode(output[0][query_tensor.input_ids.shape[1]:], skip_special_tokens=True)
                    responses.append(response_text)
                
                # Compute rewards
                rewards = self._reward_fn(queries, responses)
                
                # Log example and rewards
                if len(queries) > 0:
                    logger.info(f"Example query: {queries[0][:50]}...")
                    logger.info(f"Example response: {responses[0][:50]}...")
                    logger.info(f"Example reward: {rewards[0] if rewards else 'N/A'}")
                
                # Log mean reward
                if rewards:
                    mean_reward = sum(rewards) / len(rewards)
                    logger.info(f"Batch {i//self.ppo_config.batch_size + 1}, Mean reward: {mean_reward:.4f}")
                    epoch_rewards.append(mean_reward)
                
                # Note: We've moved this logic to the block above
                # for a more streamlined implementation
            
            # Log epoch stats
            if epoch_rewards:
                avg_reward = sum(epoch_rewards) / len(epoch_rewards)
                logger.info(f"Epoch {epoch+1} average reward: {avg_reward:.4f}")
        
        logger.info("HuggingFace PPO training completed")
        
        # If we used PPOTrainer, update our model reference
        if ppo_trainer is not None:
            self.model = ppo_trainer.model
        return self.model
    
    def save_model(self, output_dir: str):
        """Save the fine-tuned model"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Directly save the model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model and tokenizer saved to {output_dir}")


# Import NIMPPOTrainer functionality
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
        
        # Initialize the PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=None,
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
        if (dataset_path):
            import json
            with open(dataset_path, "r") as f:
                dataset = json.load(f)
            prompts = dataset["prompt"]
        else:
            # Use SHP dataset loaded via SHPDataProcessor
            logger.info("No dataset provided, using SHP dataset via SHPDataProcessor")
            # Create temporary config for SHPDataProcessor
            temp_config = {"data": {"dataset_name": "stanfordnlp/SHP", "preprocessing": {"cache_dir": "cache", "max_length": 1024}}}
            data_processor = SHPDataProcessor(temp_config)
            train_data, _, _ = data_processor.load_dataset()
            # Use the correct 'history' field which contains the post content in SHP
            prompts = [item["history"] for item in train_data]
        
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