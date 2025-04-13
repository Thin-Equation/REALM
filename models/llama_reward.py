# models/llama_reward.py
import os
import torch
import logging
from typing import Dict, List, Union, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

class LlamaRewardModel:
    """Wrapper for infly/INF-ORM-Llama3.1-70B Reward model from Hugging Face"""
    
    def __init__(
        self, 
        model_id: str, 
        quantization: Optional[str] = "4bit",
        device_map: str = "auto",
        max_length: int = 2048,
        torch_dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the Llama 3.1 70B Reward model.
        
        Args:
            model_id: Hugging Face model ID
            quantization: Quantization level ("4bit", "8bit", or None)
            device_map: Device mapping strategy
            max_length: Maximum sequence length
            torch_dtype: Torch data type for model
        """
        self.model_id = model_id
        self.max_length = max_length
        
        logger.info(f"Loading Llama 3.1 Reward model: {model_id}")
        
        # Set up quantization config if specified
        quantization_config = None
        if quantization == "4bit":
            logger.info("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            logger.info("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Determine dtype
        if torch_dtype is None:
            if torch.cuda.is_available():
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        
        # Set evaluation mode
        self.model.eval()
        logger.info("Llama 3.1 Reward model loaded successfully")
    
    def format_prompt(self, prompt: str, response: str) -> str:
        """Format prompt and response for the infly/INF-ORM-Llama3.1-70B model"""
        # Format for the infly model - adjust if the model requires a different format
        formatted_text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}"
        return formatted_text
    
    def get_reward_score(self, prompt: str, response: str, return_logits: bool = False) -> float:
        """Get reward score for a prompt-response pair"""
        try:
            # Format the input
            formatted_input = self.format_prompt(prompt, response)
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_input,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=False
            ).to(self.model.device)
            
            # Get model output
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract the reward score
            if return_logits:
                return outputs.logits.item()
            else:
                # INF-ORM models typically output a score directly
                if hasattr(outputs, "logits"):
                    score = outputs.logits.item()
                    
                    # Some reward models output a sigmoid score (0-1), others output unnormalized scores
                    # Check the range and apply sigmoid if needed
                    if abs(score) > 5:  # Heuristic for unnormalized score
                        score = torch.sigmoid(torch.tensor(score)).item()
                        
                    return float(score)
                else:
                    return outputs.rewards.item()
        
        except Exception as e:
            logger.error(f"Error getting Llama reward score: {str(e)}")
            # Return default score in case of error
            return 0.0
    
    def batch_get_reward_scores(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Get reward scores for batches of prompt-response pairs"""
        if len(prompts) != len(responses):
            raise ValueError("Number of prompts and responses must be the same")
        
        try:
            # Format the inputs
            formatted_inputs = [self.format_prompt(p, r) for p, r in zip(prompts, responses)]
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_inputs,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.model.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract the reward scores
            if hasattr(outputs, "logits"):
                scores = outputs.logits.squeeze().tolist()
                
                # Handle potentially unnormalized scores
                if not isinstance(scores, list):
                    scores = [scores]
                
                # Apply sigmoid if scores seem unnormalized
                if any(abs(s) > 5 for s in scores):
                    scores = [torch.sigmoid(torch.tensor(s)).item() for s in scores]
                
                return scores
            else:
                return outputs.rewards.squeeze().tolist()
        
        except Exception as e:
            logger.error(f"Error batch processing reward scores: {str(e)}")
            # Return default scores in case of error
            return [0.0] * len(prompts)
