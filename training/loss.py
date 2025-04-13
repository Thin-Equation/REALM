# training/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BradleyTerryLoss(nn.Module):
    """
    Bradley-Terry loss for pairwise preference learning.
    Given chosen reward and rejected reward, we want chosen > rejected.
    """
    
    def __init__(self, margin: float = 0.0):
        super(BradleyTerryLoss, self).__init__()
        self.margin = margin
    
    def forward(self, chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
        """Compute Bradley-Terry loss"""
        logits = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(logits - self.margin)
        return loss.mean()
