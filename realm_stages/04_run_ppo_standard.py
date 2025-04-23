# realm_stages/04_run_ppo_standard.py
import os
import torch
import logging
import argparse
from typing import List, Dict

# Import existing components
from config.config_loader import load_config
from models.nim_reward import NIMRewardModel
from rlhf.ppo_huggingface import HuggingFacePPOTrainer
from data.processors import SHPDataProcessor # For loading prompts
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Helper Class for Standard Reward ---

class StandardRewardPredictor:
    """
    A simple wrapper around NIMRewardModel to provide a 'predict' method
    consistent with the interface expected by HuggingFacePPOTrainer.
    """
    def __init__(self, nim_reward_model: NIMRewardModel):
        self.nim_reward_model = nim_reward_model
        logger.info("StandardRewardPredictor initialized using NIMRewardModel.")

    def predict(self, prompt: str, response: str) -> float:
        """Predict reward using ONLY the NIM reward model."""
        try:
            # Ensure this method exists and returns a float score
            return self.nim_reward_model.get_reward_score(prompt, response)
        except Exception as e:
            logger.error(f"Error getting NIM reward score for prompt='{prompt[:50]}...', response='{response[:50]}...': {e}", exc_info=True)
            # Return a neutral score or handle error as appropriate
            return 0.0

# --- Helper Function (Identical to stage 3) ---

def load_prompts(config: dict) -> List[str]:
    """Loads prompts from the SHP dataset for PPO training."""
    logger.info("Loading prompts from SHP dataset for PPO...")
    try:
        data_processor = SHPDataProcessor(config)
        train_data = data_processor.load_dataset(splits=['train'])['train']
        if 'post' in train_data.column_names:
             prompts = train_data['post']
        elif isinstance(train_data, list) and len(train_data) > 0 and isinstance(train_data[0], dict) and 'post' in train_data[0]:
             prompts = [item['post'] for item in train_data]
        else:
             logger.warning("Could not find 'post' field directly. Trying 'history'.")
             if 'history' in train_data.column_names:
                 prompts = train_data['history']
             else:
                  raise ValueError("Failed to extract prompts for PPO training. Cannot find 'post' or 'history' field.")

        ppo_config = config.get('rlhf', {}).get('ppo', {})
        max_ppo_prompts = ppo_config.get('max_prompts', None)
        if max_ppo_prompts and len(prompts) > max_ppo_prompts:
            prompts = prompts[:max_ppo_prompts]
            logger.info(f"Using subset of {max_ppo_prompts} prompts for PPO training.")
        logger.info(f"Loaded {len(prompts)} prompts for PPO.")
        return prompts
    except Exception as e:
        logger.error(f"Failed to load or process prompts: {e}", exc_info=True)
        raise

# --- Main PPO Function ---

def run_ppo_with_standard_rm(config: dict, device: torch.device):
    """
    Runs the PPO training pipeline using the Standard NIM reward model directly.

    Args:
        config: Loaded configuration dictionary.
        device: Torch device (cpu or cuda).
    """
    logger.info("--- Starting Stage 4: PPO Training with Standard NIM Reward ---")

    # 1. Load base policy model and tokenizer (same as stage 3)
    try:
        ppo_config = config['rlhf']['ppo']
        policy_model_name = ppo_config['model_name']
        logger.info(f"Loading base policy model and tokenizer: {policy_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
        if tokenizer.pad_token is None:
            logger.warning("Tokenizer does not have a pad token. Setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
        policy_model = AutoModelForCausalLM.from_pretrained(
            policy_model_name
        ).to(device)
        logger.info("Policy model and tokenizer loaded.")
    except KeyError as e:
        logger.error(f"Missing configuration key for PPO policy model: {e}. Check config['rlhf']['ppo']['model_name'].")
        raise
    except Exception as e:
        logger.error(f"Failed to load policy model or tokenizer: {e}", exc_info=True)
        raise

    # 2. Initialize Standard NIM Reward Predictor
    logger.info("Initializing Standard NIM Reward Predictor...")
    try:
        nim_reward_model = NIMRewardModel(config=config['nim_reward'], device=device)
        standard_reward_predictor = StandardRewardPredictor(nim_reward_model)
        logger.info("Standard NIM Reward Predictor initialized successfully.")
    except KeyError as e:
        logger.error(f"Missing configuration key needed for NIMRewardModel: {e}. Check config['nim_reward'].")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize Standard NIM Reward Predictor: {e}", exc_info=True)
        raise

    # 3. Prepare dataset for PPO (prompts only)
    prompts = load_prompts(config)
    ppo_dataset_dict = {"prompt": prompts}

    # 4. Initialize PPO Trainer
    logger.info("Initializing HuggingFacePPOTrainer...")
    try:
        # Pass the standard_reward_predictor instance
        ppo_trainer = HuggingFacePPOTrainer(
            config=config,
            reward_predictor=standard_reward_predictor, # Use the standard wrapper
            tokenizer=tokenizer,
            model=policy_model,
            device=device
        )
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFacePPOTrainer: {e}", exc_info=True)
        raise

    # 5. Run PPO Training
    logger.info("Starting PPO training loop...")
    try:
        num_epochs = ppo_config.get('num_epochs', 1)
        max_steps = ppo_config.get('max_steps', None)

        trained_model = ppo_trainer.train(
            dataset=ppo_dataset_dict,
            num_epochs=num_epochs,
            max_steps=max_steps
        )
    except Exception as e:
        logger.error(f"Error during PPO training: {e}", exc_info=True)
        raise

    # 6. Save the fine-tuned model (to a different directory)
    try:
        output_config = config['output']
        # Ensure a different output directory is specified in the config for the standard model
        ppo_standard_output_dir = output_config.get('ppo_standard_dir', os.path.join(output_config.get('base_dir', 'models'), 'ppo_standard_finetuned'))
        logger.info(f"Saving Standard PPO fine-tuned model to {ppo_standard_output_dir}...")
        ppo_trainer.save_model(ppo_standard_output_dir)
        logger.info(f"Model saved successfully to {ppo_standard_output_dir}.")
    except KeyError as e:
        logger.error(f"Missing configuration key for saving PPO Standard model: {e}. Check config['output']['ppo_standard_dir'].")
        raise
    except Exception as e:
        logger.error(f"Failed to save PPO Standard fine-tuned model: {e}", exc_info=True)
        raise

    logger.info("--- Stage 4: PPO Training with Standard NIM Reward Complete ---")


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Run PPO fine-tuning with Standard NIM Reward")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the main configuration file.")
    args = parser.parse_args()

    # 1. Load Configuration
    logger.info(f"Loading configuration from: {args.config}")
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration from {args.config}: {e}", exc_info=True)
        return

    # 2. Set Device
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        if not torch.cuda.is_available():
             logger.warning("CUDA not available, PPO training will run on CPU.")
    except Exception as e:
        logger.error(f"Error setting torch device: {e}", exc_info=True)
        return

    # 3. Run PPO Training with Standard RM
    try:
        run_ppo_with_standard_rm(config, device)
    except Exception as e:
        logger.error(f"PPO training with Standard RM failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
