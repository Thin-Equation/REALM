# data/processors.py
import os
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.nim_reward import BatchProcessingNimLlamaRewardModel
from utils.embedding_utils import GeminiEmbedding, cosine_similarity

logger = logging.getLogger(__name__)

class SHPDataProcessor:
    """Process the Stanford Human Preferences dataset"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dataset_name = config["data"]["dataset_name"]
        self.cache_dir = config["data"]["preprocessing"]["cache_dir"]
        self.max_length = config["data"]["preprocessing"]["max_length"]
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_dataset(self) -> Tuple[Any, Any, Any]:
        """Load the SHP dataset and return train, validation, and test splits"""
        logger.info(f"Loading dataset: {self.dataset_name}")
        
        # Load dataset using Hugging Face datasets
        dataset = load_dataset(self.dataset_name)
        
        # Process and return the dataset splits
        train_data = dataset["train"]
        val_data = dataset["validation"]
        test_data = dataset["test"]
        
        logger.info(f"Dataset loaded: {len(train_data)} training examples, "
                   f"{len(val_data)} validation examples, {len(test_data)} test examples")
        
        return train_data, val_data, test_data


class SHPRewardDataset(Dataset):
    """Dataset for training a reward model on SHP data"""
    
    def __init__(
        self, 
        data,
        nim_reward_model: BatchProcessingNimLlamaRewardModel,
        gemini_embedding: GeminiEmbedding,
        cache_dir: str,
        max_length: int = 1024,
        rebuild_cache: bool = False
    ):
        self.data = data
        self.nim_reward_model = nim_reward_model
        self.gemini_embedding = gemini_embedding
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.rebuild_cache = rebuild_cache
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def __len__(self):
        return len(self.data)
    
    def _get_cache_path(self, idx: int) -> str:
        """Get the cache file path for a data item"""
        return os.path.join(self.cache_dir, f"item_{idx}.pt")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a data item"""
        cache_path = self._get_cache_path(idx)
        
        # Use cached data if available and not rebuilding cache
        if os.path.exists(cache_path) and not self.rebuild_cache:
            try:
                return torch.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cached item {idx}: {str(e)}")
        
        # Process the data item
        item = self.data[idx]
        prompt = item["post"]
        chosen = item["preferred_comment"]
        rejected = item["dispreferred_comment"]
        
        # Truncate text if necessary
        if len(prompt) > self.max_length:
            prompt = prompt[:self.max_length]
        if len(chosen) > self.max_length:
            chosen = chosen[:self.max_length]
        if len(rejected) > self.max_length:
            rejected = rejected[:self.max_length]
        
        # Get Llama 3.1 reward scores
        chosen_llama_score = self.nim_reward_model.get_reward_score(prompt, chosen)
        rejected_llama_score = self.nim_reward_model.get_reward_score(prompt, rejected)
        
        # Get Gemini embeddings
        prompt_embedding = self.gemini_embedding.get_embedding(prompt)
        chosen_embedding = self.gemini_embedding.get_embedding(chosen)
        rejected_embedding = self.gemini_embedding.get_embedding(rejected)
        
        # Calculate similarity scores
        chosen_similarity = cosine_similarity(prompt_embedding, chosen_embedding)
        rejected_similarity = cosine_similarity(prompt_embedding, rejected_embedding)
        
        # Create feature tensors
        chosen_features = torch.tensor([chosen_llama_score, chosen_similarity], dtype=torch.float32)
        rejected_features = torch.tensor([rejected_llama_score, rejected_similarity], dtype=torch.float32)
        
        # Create result dictionary
        result = {
            "chosen_features": chosen_features,
            "rejected_features": rejected_features,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }
        
        # Cache the result
        try:
            torch.save(result, cache_path)
        except Exception as e:
            logger.warning(f"Failed to cache item {idx}: {str(e)}")
        
        return result


def create_dataloaders(
    config: Dict,
    train_data,
    val_data,
    test_data,
    nim_reward_model: BatchProcessingNimLlamaRewardModel,
    gemini_embedding: GeminiEmbedding
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Create dataloaders for training, validation, and testing"""
    batch_size = config["data"]["preprocessing"]["batch_size"]
    num_workers = config["data"]["preprocessing"]["num_workers"]
    cache_dir = config["data"]["preprocessing"]["cache_dir"]
    max_length = config["data"]["preprocessing"]["max_length"]
    
    train_dataloader, val_dataloader, test_dataloader = None, None, None
    
    # Create datasets and dataloaders if data is provided
    if train_data is not None:
        train_dataset = SHPRewardDataset(
            train_data, nim_reward_model, gemini_embedding, 
            os.path.join(cache_dir, "train"),
            max_length
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    if val_data is not None:
        val_dataset = SHPRewardDataset(
            val_data, nim_reward_model, gemini_embedding, 
            os.path.join(cache_dir, "validation"),
            max_length
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    if test_data is not None:
        test_dataset = SHPRewardDataset(
            test_data, nim_reward_model, gemini_embedding, 
            os.path.join(cache_dir, "test"),
            max_length
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_dataloader, val_dataloader, test_dataloader
