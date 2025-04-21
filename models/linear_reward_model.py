# models/linear_reward_model.py
# linear nn -> combining the embedding score and reward score from nim reward model

import torch
import torch.nn as nn
from typing import List, Optional

class LinearRewardModel(nn.Module):
    """Linear neural network for combining Llama reward and Gemini embedding similarity"""
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: List[int] = [64, 32],
        output_dim: int = 1,
        dropout: float = 0.1
    ):
        super(LinearRewardModel, self).__init__()
        
        # Build the network layers
        layers = []
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Combine all layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.model(x)
    
    def save(self, path: str) -> None:
        """Save the model to a file"""
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "input_dim": self.model[0].in_features,
                "hidden_dims": [
                    layer.out_features for layer in self.model 
                    if isinstance(layer, nn.Linear)
                ][:-1],
                "output_dim": self.model[-1].out_features,
                "dropout": next(
                    layer.p for layer in self.model 
                    if isinstance(layer, nn.Dropout)
                )
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "LinearRewardModel":
        """Load a model from a file"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model
