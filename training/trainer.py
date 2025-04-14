# training/trainer.py
import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from tqdm import tqdm
import numpy as np
import wandb

from models.reward_model import LinearRewardModel
from training.loss import BradleyTerryLoss

logger = logging.getLogger(__name__)

class RewardModelTrainer:
    """Trainer for the combined reward model"""
    
    def __init__(
        self,
        model: LinearRewardModel,
        config: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model
        self.model = model.to(self.device)
        
        # Loss function
        self.loss_fn = BradleyTerryLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        # Early stopping
        self.early_stopping_patience = config["training"]["early_stopping_patience"]
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0
        
        # Create output directory for model checkpoints
        self.output_dir = os.path.join(os.getcwd(), "models", "checkpoints")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize WandB if enabled
        if config["training"]["use_wandb"]:
            wandb.init(
                project=config["wandb"]["project"],
                entity=config["wandb"]["entity"],
                name=config["wandb"]["name"],
                config=config
            )
    
    def train(self) -> LinearRewardModel:
        """Train the model"""
        num_epochs = self.config["training"]["num_epochs"]
        logging_steps = self.config["training"]["logging_steps"]
        evaluation_steps = self.config["training"]["evaluation_steps"]
        save_steps = self.config["training"]["save_steps"]
        max_grad_norm = self.config["training"]["max_grad_norm"]
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        global_step = 0
        best_model_path = None
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            # Training loop
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for step, batch in enumerate(pbar):
                # Move batch to device
                chosen_features = batch["chosen_features"].to(self.device)
                rejected_features = batch["rejected_features"].to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                chosen_rewards = self.model(chosen_features)
                rejected_rewards = self.model(rejected_features)
                
                # Compute loss
                loss = self.loss_fn(chosen_rewards, rejected_rewards)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                
                # Calculate accuracy (chosen should have higher reward)
                predictions = (chosen_rewards > rejected_rewards).float()
                epoch_correct += predictions.sum().item()
                epoch_total += predictions.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": loss.item(),
                    "acc": epoch_correct / epoch_total if epoch_total > 0 else 0.0
                })
                
                global_step += 1
                
                # Logging
                if global_step % logging_steps == 0:
                    step_metrics = {
                        "train/loss": loss.item(),
                        "train/accuracy": epoch_correct / epoch_total if epoch_total > 0 else 0.0,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/step": global_step
                    }
                    
                    if self.config["training"]["use_wandb"]:
                        wandb.log(step_metrics)
                    
                    logger.info(f"Step {global_step}: {step_metrics}")
                
                # Evaluation
                if global_step % evaluation_steps == 0:
                    val_metrics = self.evaluate()
                    
                    # Log validation metrics
                    if self.config["training"]["use_wandb"]:
                        wandb.log(val_metrics)
                    
                    logger.info(f"Validation: {val_metrics}")
                    
                    # Check if this is the best model so far
                    if val_metrics["val/loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["val/loss"]
                        self.early_stopping_counter = 0
                        
                        # Save the best model
                        best_model_path = os.path.join(self.output_dir, "best_model.pt")
                        self.model.save(best_model_path)
                        logger.info(f"New best model saved to {best_model_path}")
                    else:
                        self.early_stopping_counter += 1
                        logger.info(f"Validation loss did not improve. Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                    
                    # Early stopping
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping triggered after {global_step} steps")
                        # Load the best model before returning
                        if best_model_path is not None and os.path.exists(best_model_path):
                            self.model = LinearRewardModel.load(best_model_path, device=self.device)
                        return self.model
                    
                    # Update learning rate scheduler
                    self.scheduler.step(val_metrics["val/loss"])
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    checkpoint_path = os.path.join(self.output_dir, f"checkpoint-{global_step}.pt")
                    self.model.save(checkpoint_path)
                    logger.info(f"Checkpoint saved to {checkpoint_path}")
            
            # End of epoch
            epoch_loss /= len(self.train_dataloader)
            epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            
            # Run evaluation at the end of each epoch
            val_metrics = self.evaluate()
            
            # Log validation metrics
            if self.config["training"]["use_wandb"]:
                wandb.log(val_metrics)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} Validation: {val_metrics}")
            
            # Check if this is the best model so far
            if val_metrics["val/loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val/loss"]
                self.early_stopping_counter = 0
                
                # Save the best model
                best_model_path = os.path.join(self.output_dir, "best_model.pt")
                self.model.save(best_model_path)
                logger.info(f"New best model saved to {best_model_path}")
            else:
                self.early_stopping_counter += 1
                logger.info(f"Validation loss did not improve. Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
            
            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Update learning rate scheduler
            self.scheduler.step(val_metrics["val/loss"])
        
        # Training complete
        logger.info("Training complete")
        
        # Test on the test set if available
        if self.test_dataloader is not None:
            # Load the best model before testing
            if best_model_path is not None and os.path.exists(best_model_path):
                self.model = LinearRewardModel.load(best_model_path, device=self.device)
            
            test_metrics = self.test()
            logger.info(f"Test results: {test_metrics}")
            
            if self.config["training"]["use_wandb"]:
                wandb.log(test_metrics)
        
        # Load the best model before returning
        if best_model_path is not None and os.path.exists(best_model_path):
            self.model = LinearRewardModel.load(best_model_path, device=self.device)
        
        return self.model
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the validation set"""
        self.model.eval()
        val_loss = 0.0
        all_chosen_rewards = []
        all_rejected_rewards = []
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                chosen_features = batch["chosen_features"].to(self.device)
                rejected_features = batch["rejected_features"].to(self.device)
                
                # Forward pass
                chosen_rewards = self.model(chosen_features)
                rejected_rewards = self.model(rejected_features)
                
                # Compute loss
                loss = self.loss_fn(chosen_rewards, rejected_rewards)
                val_loss += loss.item()
                
                # Store rewards for metrics
                all_chosen_rewards.extend(chosen_rewards.cpu().numpy())
                all_rejected_rewards.extend(rejected_rewards.cpu().numpy())
        
        # Calculate metrics
        val_loss /= len(self.val_dataloader)
        
        # Calculate accuracy
        predictions = np.array(all_chosen_rewards) > np.array(all_rejected_rewards)
        val_accuracy = np.mean(predictions)
        
        # Calculate win rate (% of cases where chosen reward > rejected reward)
        win_rate = np.mean(predictions)
        
        # Calculate margin (average difference between chosen and rejected rewards)
        margin = np.mean(np.array(all_chosen_rewards) - np.array(all_rejected_rewards))
        
        return {
            "val/loss": val_loss,
            "val/accuracy": val_accuracy,
            "val/win_rate": win_rate,
            "val/margin": margin
        }
    
    def test(self) -> Dict[str, float]:
        """Test the model on the test set"""
        self.model.eval()
        test_loss = 0.0
        all_chosen_rewards = []
        all_rejected_rewards = []
        
        with torch.no_grad():
            for batch in self.test_dataloader:
                # Move batch to device
                chosen_features = batch["chosen_features"].to(self.device)
                rejected_features = batch["rejected_features"].to(self.device)
                
                # Forward pass
                chosen_rewards = self.model(chosen_features)
                rejected_rewards = self.model(rejected_features)
                
                # Compute loss
                loss = self.loss_fn(chosen_rewards, rejected_rewards)
                test_loss += loss.item()
                
                # Store rewards for metrics
                all_chosen_rewards.extend(chosen_rewards.cpu().numpy())
                all_rejected_rewards.extend(rejected_rewards.cpu().numpy())
        
        # Calculate metrics
        test_loss /= len(self.test_dataloader)
        
        # Calculate accuracy
        predictions = np.array(all_chosen_rewards) > np.array(all_rejected_rewards)
        test_accuracy = np.mean(predictions)
        
        # Calculate win rate (% of cases where chosen reward > rejected reward)
        win_rate = np.mean(predictions)
        
        # Calculate margin (average difference between chosen and rejected rewards)
        margin = np.mean(np.array(all_chosen_rewards) - np.array(all_rejected_rewards))
        
        return {
            "test/loss": test_loss,
            "test/accuracy": test_accuracy,
            "test/win_rate": win_rate,
            "test/margin": margin
        }
