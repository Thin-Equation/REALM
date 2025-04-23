# realm_stages/02_train_realm_linear_model.py
import os
import torch
import logging
import argparse
from torch.utils.data import DataLoader, TensorDataset

# Import existing components from the repository
from config.config_loader import load_config
from models.nim_reward import NIMRewardModel
from models.linear_reward_model import LinearRewardModel
from utils.embedding_utils import LajavanessEmbedding
from data.processors import SHPDataProcessor, SHPRewardDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def train_linear_model(config: dict, device: torch.device):
    """
    Trains the REALM linear model using SHP data and existing components.

    Args:
        config: Loaded configuration dictionary.
        device: Torch device (cpu or cuda).

    Returns:
        Path to the saved linear model.
    """
    logger.info("--- Starting Stage 2: REALM Linear Model Training ---")

    # 1. Initialize base models needed for feature extraction
    logger.info("Initializing NIMRewardModel and LajavanessEmbedding...")
    try:
        nim_reward_model = NIMRewardModel(config=config['nim_reward'], device=device)
        embedding_model = LajavanessEmbedding(model_name=config['embedding']['model_id'], device=device)
    except Exception as e:
        logger.error(f"Failed to initialize NIM or Embedding model: {e}", exc_info=True)
        raise

    # 2. Load and prepare SHP dataset for reward model training
    logger.info("Loading Stanford Human Preferences (SHP) dataset...")
    try:
        data_processor = SHPDataProcessor(config)
        # Load only the training split needed for the linear model
        train_data = data_processor.load_dataset(splits=['train'])['train']
    except Exception as e:
        logger.error(f"Failed to load or process SHP dataset: {e}", exc_info=True)
        raise

    logger.info("Initializing SHPRewardDataset for feature extraction...")
    # Ensure cache directory is properly configured in config.yaml
    cache_dir_base = config.get('data', {}).get('preprocessing', {}).get('cache_dir', 'cache')
    linear_train_cache_dir = os.path.join(cache_dir_base, "linear_train")
    os.makedirs(linear_train_cache_dir, exist_ok=True)
    logger.info(f"Using cache directory for linear training features: {linear_train_cache_dir}")

    try:
        train_reward_dataset = SHPRewardDataset(
            data=train_data,
            nim_reward_model=nim_reward_model,
            embedding_model=embedding_model,
            cache_dir=linear_train_cache_dir,
            max_length=config.get('data', {}).get('preprocessing', {}).get('max_length', 512),
            rebuild_cache=config.get('data', {}).get('preprocessing', {}).get('rebuild_cache', False)
        )
    except Exception as e:
        logger.error(f"Failed to initialize SHPRewardDataset: {e}", exc_info=True)
        raise

    # 3. Prepare data specifically for the LinearRewardModel training
    logger.info("Extracting features for LinearRewardModel training...")
    all_features = []
    all_labels = []
    num_processed = 0
    num_errors = 0
    # Consider adding tqdm here for progress tracking if dataset is large
    for i in range(len(train_reward_dataset)):
        try:
            # __getitem__ should return a dict with 'chosen_features' and 'rejected_features'
            # Each feature vector should be [nim_score, similarity_score]
            item = train_reward_dataset[i]
            if 'chosen_features' in item and 'rejected_features' in item:
                all_features.append(item['chosen_features'])
                all_labels.append(1.0) # Chosen gets label 1
                all_features.append(item['rejected_features'])
                all_labels.append(0.0) # Rejected gets label 0
                num_processed += 1
            else:
                logger.warning(f"Item {i} missing 'chosen_features' or 'rejected_features'. Skipping.")
                num_errors += 1
        except Exception as e:
            logger.warning(f"Skipping item {i} due to error during feature extraction: {e}")
            num_errors += 1
            continue

    logger.info(f"Processed {num_processed} items, encountered {num_errors} errors.")
    if not all_features:
        raise ValueError("No features could be extracted. Check SHPRewardDataset logic, cache, and input data.")

    features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(1).to(device)
    logger.info(f"Created feature tensor of shape: {features_tensor.shape}")
    logger.info(f"Created label tensor of shape: {labels_tensor.shape}")


    # 4. Initialize and Train the LinearRewardModel
    logger.info("Initializing LinearRewardModel...")
    try:
        # Ensure the 'model' -> 'linear_reward' section exists in config
        linear_model_config = config['model']['linear_reward']
        linear_model = LinearRewardModel(
            input_dim=linear_model_config['input_dim'],
            hidden_dims=linear_model_config['hidden_dims'],
            output_dim=linear_model_config['output_dim'],
            dropout=linear_model_config['dropout']
        ).to(device)
    except KeyError as e:
        logger.error(f"Missing configuration key for LinearRewardModel: {e}. Check config['model']['linear_reward'].")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize LinearRewardModel: {e}", exc_info=True)
        raise


    logger.info("Training LinearRewardModel...")
    try:
        # Ensure 'training' section and relevant keys exist in config
        training_config = config['training']['linear_model']
        batch_size = training_config.get('batch_size', 32)
        learning_rate = training_config.get('learning_rate', 5e-4)
        num_epochs = training_config.get('num_epochs', 5)

        train_data = TensorDataset(features_tensor, labels_tensor)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(linear_model.parameters(), lr=learning_rate)
        # Use BCEWithLogitsLoss as it's numerically more stable than Sigmoid + BCELoss
        loss_fn = torch.nn.BCEWithLogitsLoss()

        linear_model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = linear_model(batch_features) # Shape: [batch_size, 1]
                loss = loss_fn(outputs, batch_labels) # Both should be [batch_size, 1]
                loss.backward()
                optimizer.step()

                # Track metrics
                epoch_loss += loss.item()
                # Apply sigmoid to get probabilities, then threshold
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct_predictions += (predictions == batch_labels).sum().item()
                total_predictions += batch_labels.size(0)

            avg_loss = epoch_loss / len(train_loader)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            logger.info(f"Linear Model - Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    except KeyError as e:
        logger.error(f"Missing configuration key in training section: {e}. Check config['training']['linear_model'].")
        raise
    except Exception as e:
        logger.error(f"Error during LinearRewardModel training: {e}", exc_info=True)
        raise

    # 5. Save the trained linear model
    try:
        output_config = config['output']
        linear_model_dir = output_config.get('linear_model_dir', os.path.join(output_config.get('base_dir', 'models'), 'linear_model'))
        os.makedirs(linear_model_dir, exist_ok=True)
        model_save_path = os.path.join(linear_model_dir, 'linear_reward_model.pt')
        linear_model.save(model_save_path)
        logger.info(f"Trained REALM Linear Model saved to {model_save_path}")
    except KeyError as e:
        logger.error(f"Missing configuration key in output section: {e}. Check config['output'].")
        raise
    except Exception as e:
        logger.error(f"Failed to save trained linear model: {e}", exc_info=True)
        raise

    logger.info("--- Stage 2: REALM Linear Model Training Complete ---")
    return model_save_path


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Train REALM Linear Reward Model")
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
             logger.warning("CUDA not available, training on CPU.")
    except Exception as e:
        logger.error(f"Error setting torch device: {e}", exc_info=True)
        return

    # 3. Run Training
    try:
        train_linear_model(config, device)
    except Exception as e:
        logger.error(f"Linear model training failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
