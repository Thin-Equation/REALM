# realm_together_ai.py
"""
Main script to run REALM vs Standard Reward Model PPO training pipelines
on the together.ai platform, leveraging existing repository components.
"""

import os
import torch
import logging
import yaml
import json
import argparse
from typing import Dict, Optional, Any

# Import existing components from the repository
from config.config_loader import load_config  # Assuming you have a config loader
from models.nim_reward import NIMRewardModel
from models.linear_reward_model import LinearRewardModel
from utils.embedding_utils import LajavanessEmbedding, cosine_similarity
from data.processors import SHPDataProcessor, SHPRewardDataset  # Use existing processors
from data.truthfulness_dataset import TruthfulQADataset, evaluate_model_truthfulness
from rlhf.ppo_huggingface import HuggingFacePPOTrainer
from inference.predictor import RewardPredictor # Use the existing predictor for REALM

# Import necessary Hugging Face components
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- together.ai Platform Setup (Placeholder) ---
def setup_together_ai_environment():
    """
    Configure the environment for running on the together.ai platform.
    **Action Required:** Fill this with together.ai specific setup.
    """
    logger.info("Configuring environment for together.ai platform...")

    # 1. API Keys (Load from environment variables or secure storage)
    # Example: Check if keys are set
    together_api_key = os.getenv("TOGETHER_API_KEY")
    nim_api_key = os.getenv("NVIDIA_NIM_API_KEY") # Needed for NIMRewardModel

    if not together_api_key:
        logger.warning("TOGETHER_API_KEY environment variable not set.")
        # Potentially raise an error or prompt the user
    if not nim_api_key:
        logger.warning("NVIDIA_NIM_API_KEY environment variable not set.")
        # Potentially raise an error or prompt the user

    # 2. Resource Selection (This is usually done outside the script via platform UI/config)
    logger.info("Ensure appropriate GPU resources (e.g., A100) are allocated via the together.ai platform.")

    # 3. Storage Configuration (Define paths for models, data, cache)
    # Example: Use environment variables or fixed paths expected on the platform
    os.environ["HF_HOME"] = os.getenv("HF_HOME", "/workspace/cache/huggingface")
    os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "/workspace/cache/transformers")
    # Define output directories relative to a workspace mount point
    # Example: OUTPUT_DIR = "/workspace/outputs"

    # 4. Networking/Firewall (Ensure egress is allowed if needed for APIs)
    logger.info("Verify network connectivity if external APIs (like NVIDIA NIM) are used.")

    # 5. Set other environment variables if needed
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logger.info("together.ai environment setup placeholder complete. **Review and update required.**")

# --- Helper Functions ---

class StandardRewardPredictor:
    """
    A simple wrapper around NIMRewardModel to provide a 'predict' method
    consistent with the RewardPredictor used for REALM, suitable for the PPO trainer.
    """
    def __init__(self, nim_reward_model: NIMRewardModel):
        self.nim_reward_model = nim_reward_model
        logger.info("StandardRewardPredictor initialized using NIMRewardModel.")

    def predict(self, prompt: str, response: str) -> float:
        """Predict reward using the NIM reward model."""
        # Directly call the NIM model's reward scoring method
        # Handle potential errors or default values if needed
        try:
            return self.nim_reward_model.get_reward_score(prompt, response)
        except Exception as e:
            logger.error(f"Error getting NIM reward score for prompt: '{prompt[:50]}...' response: '{response[:50]}...': {e}")
            return 0.0 # Return a default score on error


# --- Main Pipeline Functions ---

def train_realm_linear_model(config: Dict, device: torch.device) -> RewardPredictor:
    """
    Trains the REALM linear model using SHP data and existing components.

    Args:
        config: Loaded configuration dictionary.
        device: Torch device (cpu or cuda).

    Returns:
        An initialized RewardPredictor instance with the trained linear model.
    """
    logger.info("--- Starting REALM Linear Model Training ---")

    # 1. Initialize base models needed for feature extraction
    logger.info("Initializing NIMRewardModel and LajavanessEmbedding...")
    nim_reward_model = NIMRewardModel(config=config['nim_reward'], device=device)
    embedding_model = LajavanessEmbedding(model_name=config['embedding']['model_id'], device=device)

    # 2. Load and prepare SHP dataset for reward model training
    logger.info("Loading Stanford Human Preferences (SHP) dataset...")
    # Use the existing SHPDataProcessor to load raw data
    data_processor = SHPDataProcessor(config)
    train_data, val_data, _ = data_processor.load_dataset() # Load train/val splits

    logger.info("Initializing SHPRewardDataset for feature extraction...")
    # Use the SHPRewardDataset to get features (scores + similarity)
    # Note: This dataset calculates features on the fly or uses caching.
    # Ensure cache_dir in config['data']['preprocessing'] is set correctly.
    cache_dir = os.path.join(config['data']['preprocessing']['cache_dir'], "linear_train")
    train_reward_dataset = SHPRewardDataset(
        data=train_data,
        nim_reward_model=nim_reward_model,
        embedding_model=embedding_model,
        cache_dir=cache_dir,
        max_length=config['data']['preprocessing']['max_length']
        # Consider adding rebuild_cache=True if needed
    )

    # 3. Prepare data specifically for the LinearRewardModel training
    # Extract features and create labels (1 for chosen, 0 for rejected)
    logger.info("Extracting features for LinearRewardModel training...")
    all_features = []
    all_labels = []
    # Iterate through the dataset to get pre-calculated/cached features
    # This might be slow if the cache needs to be built. Consider pre-caching.
    for i in range(len(train_reward_dataset)):
        try:
            item = train_reward_dataset[i] # __getitem__ returns dict with features
            all_features.append(item['chosen_features'])
            all_labels.append(1.0)
            all_features.append(item['rejected_features'])
            all_labels.append(0.0)
        except Exception as e:
            logger.warning(f"Skipping item {i} due to error during feature extraction: {e}")
            continue # Skip problematic items

    if not all_features:
        raise ValueError("No features could be extracted. Check SHPRewardDataset and caching.")

    features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(1).to(device)

    # 4. Initialize and Train the LinearRewardModel
    logger.info("Initializing LinearRewardModel...")
    linear_model_config = config['model'] # Assuming 'model' section holds linear model params
    linear_model = LinearRewardModel(
        input_dim=linear_model_config['input_dim'],
        hidden_dims=linear_model_config['hidden_dims'],
        output_dim=linear_model_config['output_dim'],
        dropout=linear_model_config['dropout']
    ).to(device)

    logger.info("Training LinearRewardModel...")
    train_data = torch.utils.data.TensorDataset(features_tensor, labels_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config['training'].get('linear_batch_size', 32), # Add batch size to config if needed
        shuffle=True
    )

    optimizer = torch.optim.Adam(
        linear_model.parameters(),
        lr=config['training']['learning_rate'] # Use main training LR or a specific one
    )
    loss_fn = torch.nn.BCEWithLogitsLoss() # Suitable for binary preference (chosen/rejected)

    linear_model.train()
    num_epochs = config['training'].get('linear_num_epochs', 5) # Add epochs to config if needed
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_features, batch_labels in train_loader:
            outputs = linear_model(batch_features)
            loss = loss_fn(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Linear Model - Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # 5. Save the trained linear model
    output_dir = config['output'].get('linear_model_dir', 'models/trained_linear_model')
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, 'linear_reward_model.pt')
    linear_model.save(model_save_path)
    logger.info(f"Trained REALM Linear Model saved to {model_save_path}")

    # 6. Create and return the RewardPredictor instance using the trained model
    realm_predictor = RewardPredictor(
        model_path=model_save_path,
        nim_reward_model=nim_reward_model, # Re-use the initialized models
        embedding_model=embedding_model,
        device=device
    )
    logger.info("--- REALM Linear Model Training Complete ---")
    return realm_predictor


def run_ppo_pipeline(config: Dict, reward_predictor: Any, tokenizer: Any, policy_model: Any, output_dir: str, device: torch.device):
    """
    Runs the PPO training pipeline using the provided reward predictor.

    Args:
        config: Loaded configuration dictionary.
        reward_predictor: An object with a `predict(prompt, response)` method
                          (either RewardPredictor for REALM or StandardRewardPredictor).
        tokenizer: Tokenizer for the policy model.
        policy_model: Policy model (e.g., Llama-2-7B-Chat) to be fine-tuned.
        output_dir: Directory to save the PPO fine-tuned model.
        device: Torch device (cpu or cuda).
    """
    logger.info(f"--- Starting PPO Training (Output: {output_dir}) ---")

    # 1. Prepare dataset for PPO (prompts only)
    logger.info("Loading prompts from SHP dataset for PPO...")
    data_processor = SHPDataProcessor(config)
    train_data, _, _ = data_processor.load_dataset()

    # Extract prompts - Use the 'post' field from SHP as the prompt
    # Adjust field name if necessary based on SHPDataProcessor output
    if isinstance(train_data, dict) and 'post' in train_data:
         prompts = train_data['post']
    elif hasattr(train_data, 'column_names') and 'post' in train_data.column_names:
         prompts = train_data['post']
    else: # Fallback assuming it's a list of dicts
        try:
            prompts = [item['post'] for item in train_data]
        except (KeyError, TypeError) as e:
            logger.error(f"Could not extract 'post' field for prompts from train_data. Structure: {type(train_data)}. Error: {e}")
            # Attempt to use 'history' or other potential fields if applicable
            # If still failing, raise error or use a default prompt list
            raise ValueError("Failed to extract prompts for PPO training.") from e

    # Select a subset if needed (e.g., for faster testing)
    max_ppo_prompts = config['rlhf']['ppo'].get('max_prompts', None)
    if max_ppo_prompts and len(prompts) > max_ppo_prompts:
        prompts = prompts[:max_ppo_prompts]
        logger.info(f"Using subset of {max_ppo_prompts} prompts for PPO training.")

    ppo_dataset_dict = {"prompt": prompts} # PPO trainer expects a dict with prompts

    # 2. Initialize PPO Trainer (using existing HuggingFacePPOTrainer)
    logger.info("Initializing HuggingFacePPOTrainer...")
    # Note: HuggingFacePPOTrainer expects the reward_predictor passed to its constructor
    ppo_trainer = HuggingFacePPOTrainer(
        config=config, # Pass the full config dict
        reward_predictor=reward_predictor, # This needs the .predict method
        tokenizer=tokenizer,
        model=policy_model, # Pass the loaded policy model
        device=device
    )

    # 3. Run PPO Training
    logger.info("Starting PPO training loop...")
    # Use parameters from config['rlhf']['ppo']
    trained_model = ppo_trainer.train(
        dataset=ppo_dataset_dict,
        num_epochs=config['rlhf']['ppo'].get('num_epochs', 1), # Get epochs from config
        max_steps=config['rlhf']['ppo'].get('max_steps', 100) # Get max steps from config
    )

    # 4. Save the fine-tuned model
    logger.info(f"Saving PPO fine-tuned model to {output_dir}...")
    ppo_trainer.save_model(output_dir) # Use the save method from the trainer

    logger.info(f"--- PPO Training Complete (Model saved to {output_dir}) ---")


def evaluate_pipelines(config: Dict, device: torch.device) -> Dict:
    """
    Loads the PPO-trained models (REALM and Standard) and evaluates them
    on the TruthfulQA dataset.

    Args:
        config: Loaded configuration dictionary.
        device: Torch device (cpu or cuda).

    Returns:
        A dictionary containing evaluation results for both models.
    """
    logger.info("--- Starting Evaluation on TruthfulQA ---")

    # 1. Load TruthfulQA dataset
    logger.info("Loading TruthfulQA dataset...")
    truthful_qa_loader = TruthfulQADataset(
        cache_dir=os.path.join(config['data']['preprocessing']['cache_dir'], "truthfulqa")
    )
    eval_data = truthful_qa_loader.load_dataset()

    # 2. Define model paths
    realm_model_path = config['output'].get('ppo_realm_dir', 'models/ppo_realm_finetuned')
    standard_model_path = config['output'].get('ppo_standard_dir', 'models/ppo_standard_finetuned')

    results = {}

    # 3. Evaluate REALM-trained model
    logger.info(f"Evaluating REALM PPO model from: {realm_model_path}")
    try:
        tokenizer_realm = AutoTokenizer.from_pretrained(realm_model_path)
        model_realm = AutoModelForCausalLM.from_pretrained(realm_model_path)
        results["realm"] = evaluate_model_truthfulness(
            model=model_realm,
            tokenizer=tokenizer_realm,
            eval_data=eval_data,
            device=device
        )
    except Exception as e:
        logger.error(f"Failed to load or evaluate REALM model from {realm_model_path}: {e}")
        results["realm"] = {"metrics": {"accuracy": "Error", "error_message": str(e)}}

    # 4. Evaluate Standard RM-trained model
    logger.info(f"Evaluating Standard PPO model from: {standard_model_path}")
    try:
        tokenizer_standard = AutoTokenizer.from_pretrained(standard_model_path)
        model_standard = AutoModelForCausalLM.from_pretrained(standard_model_path)
        results["standard"] = evaluate_model_truthfulness(
            model=model_standard,
            tokenizer=tokenizer_standard,
            eval_data=eval_data,
            device=device
        )
    except Exception as e:
        logger.error(f"Failed to load or evaluate Standard model from {standard_model_path}: {e}")
        results["standard"] = {"metrics": {"accuracy": "Error", "error_message": str(e)}}

    # 5. Save evaluation results
    results_save_path = config['output'].get('eval_results_file', 'eval_results.json')
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
    with open(results_save_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Evaluation results saved to {results_save_path}")

    logger.info("--- Evaluation Complete ---")
    return results


# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Run REALM vs Standard RM PPO testing on together.ai")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the configuration file.")
    parser.add_argument("--run_realm", action="store_true", help="Run the full REALM pipeline (train linear model + PPO).")
    parser.add_argument("--run_standard", action="store_true", help="Run the full Standard RM pipeline (PPO only).")
    parser.add_argument("--evaluate_only", action="store_true", help="Only run evaluation on existing trained models.")
    parser.add_argument("--run_all", action="store_true", help="Run both REALM and Standard pipelines, then evaluate.")
    # Add arguments for specific steps if needed (e.g., --train_linear_only)

    args = parser.parse_args()

    # 1. Load Configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config) # Use your config loader function

    # 2. Setup Environment (Call the placeholder)
    setup_together_ai_environment() # **Requires user implementation for together.ai**

    # 3. Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Pipeline Execution ---

    if args.run_all or args.run_realm:
        logger.info("=== Starting REALM Pipeline ===")
        # a. Train the REALM Linear Model and get the predictor
        realm_reward_predictor = train_realm_linear_model(config, device)

        # b. Load base policy model and tokenizer for PPO
        logger.info(f"Loading base policy model for PPO: {config['rlhf']['ppo']['model_name']}")
        policy_tokenizer = AutoTokenizer.from_pretrained(config['rlhf']['ppo']['model_name'])
        if policy_tokenizer.pad_token is None:
            policy_tokenizer.pad_token = policy_tokenizer.eos_token
        policy_model = AutoModelForCausalLM.from_pretrained(config['rlhf']['ppo']['model_name']).to(device)

        # c. Run PPO using the REALM reward predictor
        realm_ppo_output_dir = config['output'].get('ppo_realm_dir', 'models/ppo_realm_finetuned')
        run_ppo_pipeline(
            config=config,
            reward_predictor=realm_reward_predictor,
            tokenizer=policy_tokenizer,
            policy_model=policy_model, # Pass the same loaded model instance
            output_dir=realm_ppo_output_dir,
            device=device
        )
        logger.info("=== REALM Pipeline Complete ===")


    if args.run_all or args.run_standard:
        logger.info("=== Starting Standard RM Pipeline ===")
        # a. Initialize the Standard Reward Predictor (using NIMRewardModel)
        logger.info("Initializing NIMRewardModel for Standard RM PPO...")
        nim_reward_model_standard = NIMRewardModel(config=config['nim_reward'], device=device)
        standard_reward_predictor = StandardRewardPredictor(nim_reward_model_standard)

        # b. Load base policy model and tokenizer (or reuse if run_all)
        # Avoid reloading if possible
        # If not running 'run_all', we need to load it here
        if not (args.run_all or args.run_realm): # Only load if not loaded previously
             logger.info(f"Loading base policy model for PPO: {config['rlhf']['ppo']['model_name']}")
             policy_tokenizer = AutoTokenizer.from_pretrained(config['rlhf']['ppo']['model_name'])
             if policy_tokenizer.pad_token is None:
                 policy_tokenizer.pad_token = policy_tokenizer.eos_token
             policy_model = AutoModelForCausalLM.from_pretrained(config['rlhf']['ppo']['model_name']).to(device)
        # else: # If run_all or run_realm ran, reuse the model loaded there
             # Make sure policy_model and policy_tokenizer are in scope
             # This requires careful handling of model state if pipelines modify it in-place.
             # Safest might be to reload or ensure PPO doesn't modify the base model instance directly.
             # Let's assume PPO trainer might modify, so reloading is safer if run separately.
             # Re-load for isolation if not running all sequentially in one go
             if not args.run_all:
                  logger.info(f"Re-loading base policy model for Standard PPO: {config['rlhf']['ppo']['model_name']}")
                  policy_tokenizer = AutoTokenizer.from_pretrained(config['rlhf']['ppo']['model_name'])
                  if policy_tokenizer.pad_token is None:
                      policy_tokenizer.pad_token = policy_tokenizer.eos_token
                  policy_model = AutoModelForCausalLM.from_pretrained(config['rlhf']['ppo']['model_name']).to(device)


        # c. Run PPO using the Standard NIM reward predictor
        standard_ppo_output_dir = config['output'].get('ppo_standard_dir', 'models/ppo_standard_finetuned')
        run_ppo_pipeline(
            config=config,
            reward_predictor=standard_reward_predictor,
            tokenizer=policy_tokenizer,
            policy_model=policy_model,
            output_dir=standard_ppo_output_dir,
            device=device
        )
        logger.info("=== Standard RM Pipeline Complete ===")


    if args.run_all or args.evaluate_only:
        logger.info("=== Starting Final Evaluation ===")
        eval_results = evaluate_pipelines(config, device)

        # Print a summary comparison
        logger.info("--- Evaluation Summary ---")
        try:
            realm_acc = eval_results.get("realm", {}).get("metrics", {}).get("accuracy", "N/A")
            standard_acc = eval_results.get("standard", {}).get("metrics", {}).get("accuracy", "N/A")
            logger.info(f"REALM Model Accuracy: {realm_acc}")
            logger.info(f"Standard Model Accuracy: {standard_acc}")
            if isinstance(realm_acc, (float, int)) and isinstance(standard_acc, (float, int)):
                 improvement = (realm_acc - standard_acc) * 100
                 logger.info(f"Difference (REALM - Standard): {improvement:.2f}%")
        except Exception as e:
            logger.error(f"Could not generate evaluation summary: {e}")
        logger.info("==========================")


if __name__ == "__main__":
    main()