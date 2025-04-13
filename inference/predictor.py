# inference/predictor.py
import os
import torch
import logging
from typing import Dict, Any, Optional, List, Tuple

from models.reward_model import LinearRewardModel
from models.llama_reward import LlamaRewardModel
from utils.embedding_utils import GeminiEmbedding, cosine_similarity

logger = logging.getLogger(__name__)

class RewardPredictor:
    """Predictor class for combined reward model inference"""
    
    def __init__(
        self,
        model_path: str,
        llama_reward_model: LlamaRewardModel,
        gemini_embedding: GeminiEmbedding,
        device: Optional[torch.device] = None
    ):
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = LinearRewardModel.load(model_path, device=self.device)
        self.model.eval()
        
        # Models for feature extraction
        self.llama_reward_model = llama_reward_model
        self.gemini_embedding = gemini_embedding
        
        logger.info(f"Reward predictor initialized with model from {model_path}")
    
    def predict(self, prompt: str, response: str) -> float:
        """Predict the reward for a prompt-response pair"""
        # Get Llama 3.1 reward score
        llama_score = self.llama_reward_model.get_reward_score(prompt, response)
        
        # Get Gemini embeddings
        prompt_embedding = self.gemini_embedding.get_embedding(prompt)
        response_embedding = self.gemini_embedding.get_embedding(response)
        
        # Calculate similarity score
        similarity = cosine_similarity(prompt_embedding, response_embedding)
        
        # Create feature tensor
        features = torch.tensor([llama_score, similarity], dtype=torch.float32).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            reward = self.model(features.unsqueeze(0))
        
        return reward.item()
    
    def batch_predict(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Predict rewards for a batch of prompt-response pairs"""
        if len(prompts) != len(responses):
            raise ValueError("Number of prompts and responses must be the same")
        
        rewards = []
        for prompt, response in zip(prompts, responses):
            reward = self.predict(prompt, response)
            rewards.append(reward)
        
        return rewards
    
    def compare(self, prompt: str, response1: str, response2: str) -> Tuple[float, float, int]:
        """Compare two responses for the same prompt"""
        reward1 = self.predict(prompt, response1)
        reward2 = self.predict(prompt, response2)
        
        # Return rewards and comparison result (1 if response1 is better, 2 if response2 is better)
        if reward1 > reward2:
            return reward1, reward2, 1
        else:
            return reward1, reward2, 2
