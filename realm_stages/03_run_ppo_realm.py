# realm_stages/03_run_ppo_realm.py
import os
import torch
import logging
import argparse
from typing import List, Dict

# Import existing components
from config.config_loader import load_config
from models.nim_reward import NIMRewardModel
from utils.embedding_utils import LajavanessEmbedding
from inference.predictor import RewardPredictor # The REALM predictor
from rlhf.ppo_huggingface import HuggingFacePPOTrainer
from data.processors import SHPDataProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def load_prompts(config: dict) -> List[str]:
    """Loads prompts from the SHP dataset for PPO training."""
    logger.info("Loading prompts from SHP dataset for PPO...")
    try:
        data_processor = SHPDataProcessor(config)
        # Load only the training split needed for prompts
        train_data = data_processor.load_dataset(splits=['train'])['train']

        # Extract prompts - Use the 'post' field from SHP as the prompt
        # Adjust field name if necessary based on SHPDataProcessor output structure
        if 'post' in train_data.column_names:
             prompts = train_data['post']
        elif isinstance(train_data, list) and len(train_data) > 0 and isinstance(train_data[0], dict) and 'post' in train_data[0]:
             prompts = [item['post'] for item in train_data]
        else:
             # Add fallback logic or raise error if 'post' field is not found
             # Example: Check 'history' or other potential fields
             logger.warning("Could not find 'post' field directly. Trying 'history'.")
             if 'history' in train_data.column_names:
                 prompts = train_data['history']
             else:
                  raise ValueError("Failed to extract prompts for PPO training. Cannot find 'post' or 'history' field.")

        # Select a subset if configured
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


def run_ppo_with_realm(config: dict, device: torch.device):
    """
    Runs the PPO training pipeline using the REALM reward predictor.

    Args:
        config: Loaded configuration dictionary.
        device: Torch device (cpu or cuda).
    """
    logger.info("--- Starting Stage 3: PPO Training with REALM Reward ---")

    # 1. Load base policy model and tokenizer
    try:
        ppo_config = config['rlhf']['ppo']
        policy_model_name = ppo_config['model_name']
        logger.info(f"Loading base policy model and tokenizer: {policy_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
        if tokenizer.pad_token is None:
            logger.warning("Tokenizer does not have a pad token. Setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token

        # Consider adding BitsAndBytesConfig for 4-bit/8-bit loading if needed
        # quantization_config = BitsAndBytesConfig(...)
        policy_model = AutoModelForCausalLM.from_pretrained(
            policy_model_name,
            # quantization_config=quantization_config, # If using quantization
            # torch_dtype=torch.bfloat16, # Or float16 if preferred/supported
        ).to(device)
        logger.info("Policy model and tokenizer loaded.")
    except KeyError as e:
        logger.error(f"Missing configuration key for PPO policy model: {e}. Check config['rlhf']['ppo']['model_name'].")
        raise
    except Exception as e:
        logger.error(f"Failed to load policy model or tokenizer: {e}", exc_info=True)
        raise

    # 2. Initialize REALM Reward Predictor
    logger.info("Initializing REALM Reward Predictor...")
    try:
        # Need NIM model, Embedding model, and the trained Linear Model path
        nim_reward_model = NIMRewardModel(config=config['nim_reward'], device=device)
        embedding_model = LajavanessEmbedding(model_name=config['embedding']['model_id'], device=device)

        # Get path to the trained linear model (saved in stage 2)
        output_config = config['output']
        linear_model_dir = output_config.get('linear_model_dir', os.path.join(output_config.get('base_dir', 'models'), 'linear_model'))
        linear_model_path = os.path.join(linear_model_dir, 'linear_reward_model.pt')

        if not os.path.exists(linear_model_path):
            logger.error(f"Trained linear model not found at: {linear_model_path}. Did Stage 2 run successfully?")
            raise FileNotFoundError(f"Required linear model file not found: {linear_model_path}")

        realm_predictor = RewardPredictor(
            model_path=linear_model_path,
            nim_reward_model=nim_reward_model,
            embedding_model=embedding_model,
            device=device
        )
        logger.info("REALM Reward Predictor initialized successfully.")
    except KeyError as e:
        logger.error(f"Missing configuration key needed for RewardPredictor: {e}.")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize REALM Reward Predictor: {e}", exc_info=True)
        raise

    # 3. Prepare dataset for PPO (prompts only)
    prompts = load_prompts(config)
    ppo_dataset_dict = {"prompt": prompts} # PPO trainer expects dict like {"prompt": [list_of_prompts]}

    # 4. Initialize PPO Trainer
    logger.info("Initializing HuggingFacePPOTrainer...")
    try:
        # Pass the realm_predictor instance directly
        ppo_trainer = HuggingFacePPOTrainer(
            config=config, # Pass the full config
            reward_predictor=realm_predictor, # Use the initialized REALM predictor
            tokenizer=tokenizer,
            model=policy_model, # Pass the loaded policy model
            device=device
        )
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFacePPOTrainer: {e}", exc_info=True)
        raise

    # 5. Run PPO Training
    logger.info("Starting PPO training loop...")
    try:
        # Use parameters from config['rlhf']['ppo']
        num_epochs = ppo_config.get('num_epochs', 1)
        max_steps = ppo_config.get('max_steps', None) # Allow max_steps to override epochs

        trained_model = ppo_trainer.train(
            dataset=ppo_dataset_dict,
            num_epochs=num_epochs,
            max_steps=max_steps
        )
    except Exception as e:
        logger.error(f"Error during PPO training: {e}", exc_info=True)
        raise

    # 6. Save the fine-tuned model
    try:
        ppo_realm_output_dir = output_config.get('ppo_realm_dir', os.path.join(output_config.get('base_dir', 'models'), 'ppo_realm_finetuned'))
        logger.info(f"Saving REALM PPO fine-tuned model to {ppo_realm_output_dir}...")
        ppo_trainer.save_model(ppo_realm_output_dir) # Use the save method from the trainer
        logger.info(f"Model saved successfully to {ppo_realm_output_dir}.")
    except KeyError as e:
        logger.error(f"Missing configuration key for saving PPO REALM model: {e}. Check config['output']['ppo_realm_dir'].")
        # Continue even if saving fails? Or raise? Raising seems safer.
        raise
    except Exception as e:
        logger.error(f"Failed to save PPO REALM fine-tuned model: {e}", exc_info=True)
        raise

    logger.info("--- Stage 3: PPO Training with REALM Reward Complete ---")


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Run PPO fine-tuning with REALM Reward Predictor")
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

    # 3. Run PPO Training with REALM
    try:
        run_ppo_with_realm(config, device)
    except Exception as e:
        logger.error(f"PPO training with REALM failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
