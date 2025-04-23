# realm_stages/05_evaluate_models.py
import os
import torch
import logging
import argparse
import json
from typing import Dict

# Import existing components
from config.config_loader import load_config
from data.truthfulness_dataset import TruthfulQADataset, evaluate_model_truthfulness
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def evaluate_pipeline_outputs(config: dict, device: torch.device) -> Dict:
    """
    Loads the PPO-trained models (REALM and Standard) and evaluates them
    on the TruthfulQA dataset using existing evaluation functions.

    Args:
        config: Loaded configuration dictionary.
        device: Torch device (cpu or cuda).

    Returns:
        A dictionary containing evaluation results for both models.
    """
    logger.info("--- Starting Stage 5: Evaluation on TruthfulQA ---")
    results = {}
    output_config = config['output']
    base_output_dir = output_config.get('base_dir', 'models') # Base directory for models

    # 1. Load TruthfulQA dataset
    logger.info("Loading TruthfulQA dataset...")
    try:
        # Ensure cache directory is properly configured
        cache_dir_base = config.get('data', {}).get('preprocessing', {}).get('cache_dir', 'cache')
        truthfulqa_cache_dir = os.path.join(cache_dir_base, "truthfulqa_eval")
        os.makedirs(truthfulqa_cache_dir, exist_ok=True)
        logger.info(f"Using cache directory for TruthfulQA evaluation data: {truthfulqa_cache_dir}")

        truthful_qa_loader = TruthfulQADataset(cache_dir=truthfulqa_cache_dir)
        # Load the validation split or the relevant split for evaluation
        eval_data = truthful_qa_loader.load_dataset(split='validation') # Assuming 'validation' is the target split
        logger.info(f"Loaded TruthfulQA evaluation data with {len(eval_data)} examples.")
    except Exception as e:
        logger.error(f"Failed to load TruthfulQA dataset: {e}", exc_info=True)
        raise

    # 2. Define model paths from config
    realm_model_path = output_config.get('ppo_realm_dir', os.path.join(base_output_dir, 'ppo_realm_finetuned'))
    standard_model_path = output_config.get('ppo_standard_dir', os.path.join(base_output_dir, 'ppo_standard_finetuned'))

    # 3. Evaluate REALM-trained model
    logger.info(f"Evaluating REALM PPO model from: {realm_model_path}")
    if os.path.exists(realm_model_path):
        try:
            tokenizer_realm = AutoTokenizer.from_pretrained(realm_model_path)
            model_realm = AutoModelForCausalLM.from_pretrained(realm_model_path).to(device)
            # Ensure evaluate_model_truthfulness takes the correct args and returns desired structure
            realm_eval_results = evaluate_model_truthfulness(
                model=model_realm,
                tokenizer=tokenizer_realm,
                eval_data=eval_data,
                device=device,
                config=config.get('evaluation', {}) # Pass evaluation specific config if needed
            )
            results["realm"] = realm_eval_results
            logger.info(f"REALM Model Evaluation Results: {realm_eval_results.get('metrics', 'N/A')}")
            # Clean up model to free memory if running sequentially
            del model_realm
            del tokenizer_realm
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Failed to load or evaluate REALM model from {realm_model_path}: {e}", exc_info=True)
            results["realm"] = {"error": str(e), "metrics": {"accuracy": "Error"}}
    else:
        logger.warning(f"REALM model directory not found: {realm_model_path}. Skipping evaluation.")
        results["realm"] = {"error": "Model directory not found.", "metrics": {"accuracy": "Skipped"}}


    # 4. Evaluate Standard RM-trained model
    logger.info(f"Evaluating Standard PPO model from: {standard_model_path}")
    if os.path.exists(standard_model_path):
        try:
            tokenizer_standard = AutoTokenizer.from_pretrained(standard_model_path)
            model_standard = AutoModelForCausalLM.from_pretrained(standard_model_path).to(device)
            standard_eval_results = evaluate_model_truthfulness(
                model=model_standard,
                tokenizer=tokenizer_standard,
                eval_data=eval_data,
                device=device,
                config=config.get('evaluation', {})
            )
            results["standard"] = standard_eval_results
            logger.info(f"Standard Model Evaluation Results: {standard_eval_results.get('metrics', 'N/A')}")
             # Clean up model
            del model_standard
            del tokenizer_standard
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Failed to load or evaluate Standard model from {standard_model_path}: {e}", exc_info=True)
            results["standard"] = {"error": str(e), "metrics": {"accuracy": "Error"}}
    else:
        logger.warning(f"Standard model directory not found: {standard_model_path}. Skipping evaluation.")
        results["standard"] = {"error": "Model directory not found.", "metrics": {"accuracy": "Skipped"}}


    # 5. Save evaluation results
    try:
        results_save_path = output_config.get('eval_results_file', os.path.join(base_output_dir, 'evaluation_results.json'))
        # Ensure the directory exists
        os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
        with open(results_save_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Evaluation results saved to {results_save_path}")
    except KeyError as e:
        logger.error(f"Missing configuration key for saving evaluation results: {e}. Check config['output']['eval_results_file'].")
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {e}", exc_info=True)


    logger.info("--- Stage 5: Evaluation Complete ---")
    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 5: Evaluate REALM and Standard PPO Models")
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
        # Evaluation might require less memory, but GPU is still preferred
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
    except Exception as e:
        logger.error(f"Error setting torch device: {e}", exc_info=True)
        return

    # 3. Run Evaluation
    try:
        final_results = evaluate_pipeline_outputs(config, device)
        # Optionally print a summary comparison here
        realm_acc = final_results.get("realm", {}).get("metrics", {}).get("accuracy", "N/A")
        std_acc = final_results.get("standard", {}).get("metrics", {}).get("accuracy", "N/A")
        logger.info("="*20 + " Final Summary " + "="*20)
        logger.info(f"REALM Model Accuracy: {realm_acc}")
        logger.info(f"Standard Model Accuracy: {std_acc}")
        logger.info("="*55)

    except Exception as e:
        logger.error(f"Evaluation process failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
