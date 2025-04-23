# main.py
import os
import argparse
import logging
import yaml
import torch
import random
import numpy as np
import json # Added for JSON serialization
import matplotlib.pyplot as plt # Added for plotting
import sys
from typing import Dict, Any, Optional # Added List and Optional for typing
from dotenv import load_dotenv
from tqdm import tqdm # Added for progress bars

# First import validation to ensure it's available for early checking
from utils.validation import validate_environment, validate_model_file

# Then import other modules
from utils.embedding_utils import LajavanessEmbedding
from data.processors import SHPDataProcessor, create_dataloaders, TruthfulQAProcessor
from models.reward_model import LinearRewardModel
from training.trainer import RewardModelTrainer
from inference.predictor import RewardPredictor, NIMRewardAdapter
from rlhf.ppo_integration import HuggingFacePPOTrainer
from models.nim_reward import NIMRewardModel
from transformers import AutoModelForCausalLM, AutoTokenizer # Added for loading models and tokenizers

# Load environment variables first thing
load_dotenv()

# --- Utility functions for model verification ---
def get_model_size(path: str) -> float:
    """Get the size of a model file or directory in MB"""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    elif os.path.isdir(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)
    return 0

# --- ModelEvaluator class for model evaluation ---
class ModelEvaluator:
    """Class to evaluate fine-tuned LLMs on TruthfulQA"""
    
    def __init__(
        self,
        combined_model_path: str,
        nim_model_path: str,
        base_model_name: str = "meta-llama/Meta-Llama-3.1-Instruct-8B",
        device: Optional[torch.device] = None,
        logger = None
    ):
        self.device = device if device else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.logger = logger if logger else logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")
        
        # Model paths
        self.combined_model_path = combined_model_path
        self.nim_model_path = nim_model_path
        self.base_model_name = base_model_name
        
        # Store results
        self.results = {
            "combined": [],
            "nim": [],
        }
        
        # Initialize TruthfulQA processor
        self.truthfulqa_processor = TruthfulQAProcessor()
        
        # Validate models before loading
        self._validate_models()
        
        # Load models and tokenizers
        self._load_models()
    
    def _validate_models(self) -> None:
        """Validate model files before loading"""
        self.logger.info("Validating model files...")
        
        # Validate combined model
        is_valid, message = validate_model_file(self.combined_model_path, "Combined-reward fine-tuned model")
        if not is_valid:
            self.logger.error(f"Combined model validation failed: {message}")
            raise ValueError(f"Combined model validation failed: {message}")
        else:
            self.logger.info(f"✓ Combined model validated: {message}")
            
        # Validate NIM model
        is_valid, message = validate_model_file(self.nim_model_path, "NIM-reward fine-tuned model")
        if not is_valid:
            self.logger.error(f"NIM model validation failed: {message}")
            raise ValueError(f"NIM model validation failed: {message}")
        else:
            self.logger.info(f"✓ NIM model validated: {message}")
    
    def _load_models(self) -> None:
        """Load all models and tokenizers"""
        self.logger.info("Loading models and tokenizers...")
        
        # Load combined-reward model
        self.logger.info(f"Loading combined-reward model from {self.combined_model_path}")
        self.combined_model = AutoModelForCausalLM.from_pretrained(self.combined_model_path)
        self.combined_model.to(self.device)
        self.combined_model.eval()
        
        # Load NIM-reward model
        self.logger.info(f"Loading NIM-reward model from {self.nim_model_path}")
        self.nim_model = AutoModelForCausalLM.from_pretrained(self.nim_model_path)
        self.nim_model.to(self.device)
        self.nim_model.eval()
        
        # Load tokenizer (shared)
        self.logger.info(f"Loading tokenizer from {self.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.logger.info("All models and tokenizers loaded successfully")
    
    def generate_answer(self, model, prompt: str, max_length: int = 512) -> str:
        """Generate answer from model"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate with model
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part (remove the prompt)
        answer = response[len(prompt):]
        
        return answer.strip()
    
    def evaluate_on_truthfulqa(self) -> Dict[str, Any]:
        """Evaluate both models on TruthfulQA dataset"""
        # Load TruthfulQA dataset using the processor
        self.logger.info("Loading TruthfulQA dataset...")
        eval_data = self.truthfulqa_processor.load_dataset(split="validation")
        
        self.logger.info(f"Evaluating models on {len(eval_data)} TruthfulQA examples")
        
        for i, example in enumerate(tqdm(eval_data)):
            question = example["question"]
            correct_answers = example["correct_answers"]
            incorrect_answers = example["incorrect_answers"]
            
            # Format as a prompt
            prompt = f"Question: {question}\nAnswer: "
            
            # Generate answers with both models
            combined_answer = self.generate_answer(self.combined_model, prompt)
            nim_answer = self.generate_answer(self.nim_model, prompt)
            
            # Evaluate truthfulness using the processor
            combined_eval = self.truthfulqa_processor.evaluate_truthfulness(
                question, combined_answer, correct_answers, incorrect_answers
            )
            
            # Evaluate truthfulness for NIM model
            nim_eval = self.truthfulqa_processor.evaluate_truthfulness(
                question, nim_answer, correct_answers, incorrect_answers
            )
            
            # Store results
            self.results["combined"].append({
                "question": question,
                "answer": combined_answer,
                "metrics": combined_eval
            })
            
            self.results["nim"].append({
                "question": question,
                "answer": nim_answer,
                "metrics": nim_eval
            })
            
            # Log progress every 10 items
            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{len(eval_data)} questions")
        
        # Calculate and return overall metrics using the processor
        return self.truthfulqa_processor.calculate_metrics(self.results)
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate overall metrics from evaluation results"""
        return self.truthfulqa_processor.calculate_metrics(self.results)
    
    def save_results(self, output_dir: str) -> None:
        """Save evaluation results and metrics"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        with open(os.path.join(output_dir, "detailed_results.json"), "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Save metrics
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Create comparison visualization
        self._create_comparison_plot(metrics, output_dir)
        
        self.logger.info(f"Results saved to {output_dir}")
        
        # Print summary
        self._print_summary(metrics)
    
    def _print_summary(self, metrics: Dict[str, Any]) -> None:
        """Print summary of evaluation results"""
        self.logger.info("\n---- EVALUATION SUMMARY ----")
        self.logger.info(f"Total questions evaluated: {metrics['combined']['total_questions']}")
        
        self.logger.info("\nCombined Reward Model metrics:")
        self.logger.info(f"  Truthfulness: {metrics['combined']['truthfulness'] * 100:.2f}%")
        self.logger.info(f"  Exact Match: {metrics['combined']['exact_match'] * 100:.2f}%")
        self.logger.info(f"  Contains Correct: {metrics['combined']['contains_correct'] * 100:.2f}%")
        self.logger.info(f"  Contains Incorrect: {metrics['combined']['contains_incorrect'] * 100:.2f}%")
        
        self.logger.info("\nNIM Reward Model metrics:")
        self.logger.info(f"  Truthfulness: {metrics['nim']['truthfulness'] * 100:.2f}%")
        self.logger.info(f"  Exact Match: {metrics['nim']['exact_match'] * 100:.2f}%")
        self.logger.info(f"  Contains Correct: {metrics['nim']['contains_correct'] * 100:.2f}%")
        self.logger.info(f"  Contains Incorrect: {metrics['nim']['contains_incorrect'] * 100:.2f}%")
    
    def _create_comparison_plot(self, metrics: Dict[str, Any], output_dir: str) -> None:
        """Create comparison plot of metrics"""
        # Extract metrics to plot
        metric_names = ["truthfulness", "exact_match", "contains_correct", "contains_incorrect"]
        combined_values = [metrics["combined"][metric] for metric in metric_names]
        nim_values = [metrics["nim"][metric] for metric in metric_names]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        x = np.arange(len(metric_names))
        width = 0.35
        
        plt.bar(x - width/2, [val * 100 for val in combined_values], width, label="Combined Reward Model")
        plt.bar(x + width/2, [val * 100 for val in nim_values], width, label="NIM Reward Model")
        
        plt.xlabel("Metrics")
        plt.ylabel("Percentage (%)")
        plt.title("Comparison of Model Performance on TruthfulQA")
        plt.xticks(x, [metric.replace("_", " ").title() for metric in metric_names])
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "metrics_comparison.png"))
        plt.close()

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
    """Main entry point with comprehensive validation before execution"""
    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description="Combined Reward Model for RLHF")
        parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
        parser.add_argument("--mode", type=str, choices=["train", "ppo", "dpo", "predict", "nim_ppo", "verify", "evaluate"], default="train", help="Mode to run")
        parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint (required for ppo, dpo, predict)")
        parser.add_argument("--prompt", type=str, default=None, help="Prompt for prediction (used in predict mode)")
        parser.add_argument("--response", type=str, default=None, help="Response for prediction (used in predict mode)")
        parser.add_argument("--dataset_path", type=str, default=None, help="Path to dataset for RLHF (used in ppo/dpo/nim_ppo mode)")
        parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the fine-tuned model (used in ppo/nim_ppo mode) or evaluation results (used in evaluate mode)")
        parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of training steps (used in ppo/nim_ppo mode)")
        parser.add_argument("--reward_model_path", type=str, default="models/final_model.pt", help="Path for reward model verification")
        parser.add_argument("--combined_model_path", type=str, default="models/ppo_finetuned", help="Path for combined model verification/evaluation")
        parser.add_argument("--nim_model_path", type=str, default="models/nim_ppo_finetuned", help="Path for NIM model verification/evaluation")
        parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-Instruct-8B", help="Base model name for evaluation")
        parser.add_argument("--skip_validation", action="store_true", help="Skip environment validation (not recommended)")
        
        args = parser.parse_args()
        
        # Setup logging before anything else
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        
        logger.info(f"Starting Combined Reward Model in {args.mode} mode")
        
        # Validate environment first with the config path
        if not args.skip_validation:
            logger.info(f"Validating environment using config: {args.config}")
            validate_environment(args.config)
            logger.info("Environment validation passed successfully")
        else:
            logger.warning("Environment validation skipped by user request. This is not recommended.")
        
        # Load configuration after validation
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Set random seed
        if "training" in config and "seed" in config["training"]:
            seed = config["training"]["seed"]
            logger.info(f"Setting random seed to {seed} for reproducibility")
            set_seed(seed)
        else:
            logger.warning("No random seed specified in config, using system default")
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize models (only if needed for the mode)
        nim_reward_model = None
        embedding_model = None
        
        if args.mode in ["train", "predict", "ppo", "nim_ppo"]:
            # Initialize Llama 3.1 Nemotron Reward model via NIM API
            logger.info("Initializing NIM reward model...")
            nim_reward_model = NIMRewardModel(
                api_key=config["nim_reward"]["api_key"],
                base_url=config["nim_reward"]["base_url"],
                model_id=config["nim_reward"]["model_id"],
                max_retries=config["nim_reward"]["max_retries"],
                retry_delay=config["nim_reward"]["retry_delay"]
            )
            
            # Initialize Lajavaness Embedding
            logger.info("Initializing embedding model...")
            embedding_model = LajavanessEmbedding(
                model_id=config["embedding"]["model_id"]
            )
        
        # --- Execute the requested mode ---
        if args.mode == "train":
            # Load and process dataset
            logger.info("Loading and processing dataset...")
            data_processor = SHPDataProcessor(config)
            train_data, val_data, test_data = data_processor.load_dataset()
            
            # Create dataloaders
            logger.info("Creating dataloaders...")
            train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
                config, train_data, val_data, test_data, nim_reward_model, embedding_model
            )
            
            # Initialize model
            logger.info("Initializing model...")
            model = LinearRewardModel(
                input_dim=config["model"]["input_dim"],
                hidden_dims=config["model"]["hidden_dims"],
                output_dim=config["model"]["output_dim"],
                dropout=config["model"]["dropout"]
            )
            
            # Initialize trainer
            logger.info("Setting up trainer...")
            trainer = RewardModelTrainer(
                model=model,
                config=config,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                device=device
            )
            
            # Train model
            logger.info("Starting training...")
            trained_model = trainer.train()
            
            # Save final model
            final_model_path = os.path.join("models", "final_model.pt")
            logger.info(f"Saving final model to {final_model_path}")
            trained_model.save(final_model_path)
            logger.info(f"Final model saved to {final_model_path}")
            
        elif args.mode == "predict":
            # Check if model path, prompt, and response are provided
            if args.model_path is None:
                raise ValueError("Model path must be provided for predict mode")
            if args.prompt is None or args.response is None:
                raise ValueError("Prompt and response must be provided for predict mode")
            
            # Validate model file exists
            is_valid, message = validate_model_file(args.model_path, "Reward model")
            if not is_valid:
                raise ValueError(f"Invalid model file: {message}")
            
            # Initialize predictor
            logger.info(f"Initializing predictor with model: {args.model_path}")
            predictor = RewardPredictor(
                model_path=args.model_path,
                nim_reward_model=nim_reward_model,
                embedding_model=embedding_model,
                device=device
            )
            
            # Predict reward
            logger.info("Predicting reward...")
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
            
            # Use Hugging Face's PPO Trainer implementation
            logger.info("Using HuggingFace's PPO implementation")
            ppo_trainer = HuggingFacePPOTrainer(
                config=config,
                reward_predictor=predictor,
                device=device
            )
            
            # Train with PPO using direct dataset loading
            max_steps = config["rlhf"]["ppo"].get("max_steps", 100)
            num_epochs = config["rlhf"]["ppo"].get("num_epochs", 1)
            
            # If custom dataset path is provided, use it
            if args.dataset_path:
                import json
                with open(args.dataset_path, "r") as f:
                    dataset = json.load(f)
                logger.info(f"Training with custom dataset from {args.dataset_path}")
                ppo_trainer.train(
                    dataset=dataset,
                    num_epochs=num_epochs,
                    max_steps=max_steps
                )
            else:
                # Otherwise use SHP dataset directly
                logger.info("Training with SHP dataset via SHPDataProcessor")
                # Create a temporary SHP dataset loader
                temp_data_processor = SHPDataProcessor(config)
                train_data, _, _ = temp_data_processor.load_dataset()
                # Create dataset in the expected format for PPO training
                dataset = {
                    "prompt": [item["history"] for item in train_data]
                }
                ppo_trainer.train(
                    dataset=dataset,
                    num_epochs=num_epochs,
                    max_steps=max_steps
                )
            
            # Save the fine-tuned model
            output_dir = args.output_dir if args.output_dir else os.path.join("models", "ppo_finetuned")
            ppo_trainer.save_model(output_dir)
            logger.info(f"PPO fine-tuned model saved to {output_dir}")
        
        elif args.mode == "nim_ppo":
            # Initialize NIM reward adapter for direct NIM reward usage
            nim_reward_adapter = NIMRewardAdapter(nim_reward_model)
            
            # Initialize PPO trainer with NIM reward adapter
            ppo_trainer = HuggingFacePPOTrainer(
                config=config,
                reward_predictor=nim_reward_adapter,  # Use adapter with the same interface as RewardPredictor
                device=device
            )
            
            # Train with PPO using direct dataset loading
            max_steps = args.max_steps or config["rlhf"]["ppo"].get("max_steps", 1000)
            num_epochs = config["rlhf"]["ppo"].get("num_epochs", 1)
            
            # If custom dataset path is provided, use it
            if args.dataset_path:
                import json
                with open(args.dataset_path, "r") as f:
                    dataset = json.load(f)
                logger.info(f"Training with custom dataset from {args.dataset_path}")
                ppo_trainer.train(
                    dataset=dataset,
                    num_epochs=num_epochs,
                    max_steps=max_steps
                )
            else:
                # Otherwise use SHP dataset directly
                logger.info("Training with SHP dataset via SHPDataProcessor")
                # Create a temporary SHP dataset loader
                temp_data_processor = SHPDataProcessor(config)
                train_data, _, _ = temp_data_processor.load_dataset()
                # Create dataset in the expected format for PPO training
                dataset = {
                    "prompt": [item["history"] for item in train_data]
                }
                ppo_trainer.train(
                    dataset=dataset,
                    num_epochs=num_epochs,
                    max_steps=max_steps
                )
            
            # Save the fine-tuned model
            output_dir = args.output_dir if args.output_dir else os.path.join("models", "nim_ppo_finetuned")
            ppo_trainer.save_model(output_dir)
            logger.info(f"NIM PPO fine-tuned model saved to {output_dir}")

        elif args.mode == "verify":
            logger.info("Running model verification...")
            
            # Check reward model
            reward_model_ok = validate_model_file(args.reward_model_path, "Reward model (LinearRewardModel)", logger)
            if reward_model_ok:
                size_mb = get_model_size(args.reward_model_path)
                logger.info(f"  Reward model size: {size_mb:.2f} MB")
            
            # Check combined-reward fine-tuned model
            combined_model_ok = validate_model_file(args.combined_model_path, "Combined-reward fine-tuned model", logger)
            if combined_model_ok:
                size_mb = get_model_size(args.combined_model_path)
                logger.info(f"  Combined-reward fine-tuned model size: {size_mb:.2f} MB")
            
            # Check NIM-reward fine-tuned model
            nim_model_ok = validate_model_file(args.nim_model_path, "NIM-reward fine-tuned model", logger)
            if nim_model_ok:
                size_mb = get_model_size(args.nim_model_path)
                logger.info(f"  NIM-reward fine-tuned model size: {size_mb:.2f} MB")
            
            # Overall status
            all_models_ok = reward_model_ok and combined_model_ok and nim_model_ok
            
            if all_models_ok:
                logger.info("✅ All three required model weights are saved correctly!")
                logger.info("  1. Reward Model (LinearRewardModel): " + args.reward_model_path)
                logger.info("  2. Combined-Reward Fine-tuned LLM: " + args.combined_model_path)
                logger.info("  3. NIM-Reward Fine-tuned LLM: " + args.nim_model_path)
                
                # Remind about downloading models from Brev
                logger.info("\nTo download these models from your Brev instance, run:")
                logger.info(f"  brev pull instance-name:/app/{args.reward_model_path} ./local-path/")
                logger.info(f"  brev pull -r instance-name:/app/{args.combined_model_path} ./local-path/combined/")
                logger.info(f"  brev pull -r instance-name:/app/{args.nim_model_path} ./local-path/nim/")
            else:
                logger.error("❌ Some required model weights are missing!")
                if not reward_model_ok:
                    logger.error("  - Reward model is missing. Run: python main.py --mode train")
                if not combined_model_ok:
                    logger.error("  - Combined-reward fine-tuned model is missing. Run: python main.py --mode ppo --model_path models/final_model.pt")
                if not nim_model_ok:
                    logger.error("  - NIM-reward fine-tuned model is missing. Run: python main.py --mode nim_ppo")
            
            logger.info("Model verification completed.")

        elif args.mode == "evaluate":
            logger.info("Running model evaluation...")
            
            # Set output directory
            output_dir = args.output_dir if args.output_dir else "evaluation_results"
            
            # Add argument for base model
            base_model = args.base_model if hasattr(args, "base_model") else "meta-llama/Meta-Llama-3.1-Instruct-8B"
            
            # Create evaluator
            evaluator = ModelEvaluator(
                combined_model_path=args.combined_model_path,
                nim_model_path=args.nim_model_path,
                base_model_name=base_model,
                device=device,
                logger=logger
            )
            
            # Evaluate on TruthfulQA
            evaluator.evaluate_on_truthfulqa()
            
            # Save results
            evaluator.save_results(output_dir)
            
            logger.info("Model evaluation completed.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

