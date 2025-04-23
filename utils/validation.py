# utils/validation.py
import os
import logging
import importlib
import torch
import shutil
import yaml
import sys
from typing import List, Tuple

logger = logging.getLogger(__name__)

def validate_environment(config_path: str = "config/config.yaml") -> bool:
    """
    Validate environment, dependencies, configuration files, and directories
    before running the application.
    
    Args:
        config_path: Path to the configuration file to validate
        
    Returns:
        bool: True if all validation checks pass, raises an error otherwise
    """
    checks = []
    warnings = []
    
    logger.info("Running comprehensive environment validation...")
    
    # Section 1: Validate API keys
    logger.info("Checking required API keys...")
    api_keys = _check_api_keys()
    checks.extend(api_keys[0])
    warnings.extend(api_keys[1])
    
    # Section 2: Validate configuration file
    logger.info("Checking configuration files...")
    config_checks = _validate_config(config_path)
    checks.extend(config_checks[0])
    warnings.extend(config_checks[1])
    
    # Section 3: Check required packages
    logger.info("Checking required Python packages...")
    package_checks = _check_packages()
    checks.extend(package_checks[0])
    warnings.extend(package_checks[1])
    
    # Section 4: Check hardware resources
    logger.info("Checking hardware resources...")
    hw_checks = _check_hardware()
    checks.extend(hw_checks[0])
    warnings.extend(hw_checks[1])
    
    # Section 5: Check directories and permissions
    logger.info("Checking directories and permissions...")
    dir_checks = _check_directories()
    checks.extend(dir_checks[0])
    warnings.extend(dir_checks[1])
    
    # Section 6: Check for model files
    logger.info("Checking model files...")
    model_checks = _check_models()
    checks.extend(model_checks[0])
    warnings.extend(model_checks[1])
    
    # Output warnings
    if warnings:
        logger.warning("⚠️ Environment validation warnings:")
        for warning in warnings:
            logger.warning(f"  ! {warning}")
    
    # If any checks failed, raise error
    if checks:
        error_msg = "❌ Environment validation failed:\n" + "\n".join(checks)
        logger.error(error_msg)
        raise EnvironmentError(error_msg)
    
    logger.info("✅ All environment checks passed successfully")
    return True

def _check_api_keys() -> Tuple[List[str], List[str]]:
    """Check for required API keys in the environment"""
    errors = []
    warnings = []
    
    # Required API keys
    if not os.environ.get("NVIDIA_NIM_API_KEY"):
        errors.append("NVIDIA_NIM_API_KEY is not set in environment")
    
    # Optional API keys
    optional_keys = {
        "WANDB_API_KEY": "Weights & Biases logging",
        "HUGGINGFACE_API_KEY": "Hugging Face model hub access",
        "OPENAI_API_KEY": "OpenAI API access"
    }
    
    for key, purpose in optional_keys.items():
        if not os.environ.get(key):
            warnings.append(f"{key} is not set. This may limit {purpose}.")
    
    return errors, warnings

def _validate_config(config_path: str) -> Tuple[List[str], List[str]]:
    """Validate the configuration file structure and required fields"""
    errors = []
    warnings = []
    
    if not os.path.exists(config_path):
        errors.append(f"Configuration file not found: {config_path}")
        return errors, warnings
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Check required config sections
        required_sections = ["model", "data", "training", "rlhf", "nim_reward", "embedding"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required config section: {section}")
        
        # Check specific critical config values if their sections exist
        if "model" in config:
            model_config = config["model"]
            for field in ["input_dim", "hidden_dims", "output_dim"]:
                if field not in model_config:
                    errors.append(f"Missing required model config field: {field}")
        
        if "data" in config:
            if "dataset_name" not in config["data"]:
                warnings.append("No dataset_name specified in data config")
        
        if "training" in config:
            training_config = config["training"]
            for field in ["batch_size", "learning_rate", "num_epochs"]:
                if field not in training_config:
                    warnings.append(f"Missing recommended training config field: {field}")
        
        if "rlhf" in config and "ppo" in config["rlhf"]:
            ppo_config = config["rlhf"]["ppo"]
            for field in ["model_name", "learning_rate", "batch_size"]:
                if field not in ppo_config:
                    errors.append(f"Missing required PPO config field: {field}")
        
        if "nim_reward" in config:
            nim_config = config["nim_reward"]
            for field in ["api_key", "base_url", "model_id"]:
                if field not in nim_config:
                    errors.append(f"Missing required NIM reward config field: {field}")
        
    except Exception as e:
        errors.append(f"Error parsing config file: {str(e)}")
    
    return errors, warnings

def _check_packages() -> Tuple[List[str], List[str]]:
    """Check for required and recommended Python packages"""
    errors = []
    warnings = []
    
    # Required packages
    required_packages = {
        "torch": "torch",
        "transformers": "transformers",
        "datasets": "datasets",
        "trl": "trl",
        "peft": "peft",
        "numpy": "numpy",
        "fastapi": "fastapi",
        "wandb": "wandb"
    }
    
    # Optional packages with functionality they enable
    optional_packages = {
        "openai": "OpenAI API integration",
        "google.generativeai": "Google Generative AI integration",
        "matplotlib": "Data visualization",
        "pandas": "Data manipulation",
        "sklearn": "Machine learning utilities"
    }
    
    # Check required packages
    for import_name, package_name in required_packages.items():
        try:
            module = importlib.import_module(import_name)
            logger.info(f"✓ Required package '{package_name}' is installed")
            
            # Check specific package versions
            if hasattr(module, "__version__"):
                version = module.__version__
                logger.info(f"  Version: {version}")
                
                # Version-specific checks
                if import_name == "trl" and version < "0.4.0":
                    warnings.append(f"TRL version {version} may be outdated. Version 0.4.0+ recommended.")
                elif import_name == "transformers" and version < "4.30.0":
                    warnings.append(f"Transformers version {version} may be outdated. Version 4.30.0+ recommended.")
                    
        except ImportError:
            errors.append(f"Required package '{package_name}' is not installed")
    
    # Check optional packages
    for import_name, feature in optional_packages.items():
        try:
            importlib.import_module(import_name)
            logger.info(f"✓ Optional package for {feature} is installed")
        except ImportError:
            warnings.append(f"Optional package for {feature} is not installed")
    
    return errors, warnings

def _check_hardware() -> Tuple[List[str], List[str]]:
    """Check hardware capabilities and resources"""
    errors = []
    warnings = []
    
    # Check CUDA availability
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_capability = torch.cuda.get_device_capability(0)
            cuda_version = torch.version.cuda
            
            logger.info(f"✓ CUDA is available: {device_name}")
            logger.info(f"  CUDA capability: {device_capability[0]}.{device_capability[1]}")
            logger.info(f"  CUDA version: {cuda_version}")
            
            # Check compute capability
            if device_capability[0] < 3:
                warnings.append(f"GPU compute capability {device_capability[0]}.{device_capability[1]} may be too low for optimal performance")
                
            # Check GPU memory
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"  GPU memory: {gpu_memory:.2f} GB")
                
                if gpu_memory < 8:
                    warnings.append(f"GPU has only {gpu_memory:.2f} GB of memory, which may be insufficient for larger models")
            except Exception as e:
                warnings.append(f"Could not determine GPU memory: {e}")
                
        else:
            warnings.append("CUDA is not available. Using CPU will be significantly slower for training")
            logger.warning("⚠️ CUDA is not available, using CPU (this will be slow)")
    except Exception as e:
        warnings.append(f"Error checking CUDA availability: {e}")
    
    # Check disk space
    try:
        _, _, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        logger.info(f"✓ Available disk space: {free_gb:.2f} GB")
        
        if free_gb < 10:
            warnings.append(f"Only {free_gb:.2f} GB of disk space available. At least 10 GB recommended.")
    except Exception as e:
        warnings.append(f"Could not check disk space: {e}")
    
    # Check RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        logger.info(f"✓ System RAM: {ram_gb:.2f} GB")
        
        if ram_gb < 16:
            warnings.append(f"System has {ram_gb:.2f} GB of RAM. At least 16 GB recommended.")
    except ImportError:
        warnings.append("psutil module not installed. Could not check system RAM.")
    except Exception as e:
        warnings.append(f"Error checking system RAM: {e}")
    
    return errors, warnings

def _check_directories() -> Tuple[List[str], List[str]]:
    """Check directories for existence and proper permissions"""
    errors = []
    warnings = []
    
    # Define directories to check with descriptions
    directories = {
        "cache": "Cache directory for dataset and embedding storage",
        "models": "Model storage directory",
        "models/checkpoints": "Model checkpoints directory",
        "config": "Configuration directory",
        "data": "Data processing directory"
    }
    
    # Check each directory
    for dir_path, description in directories.items():
        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        
        # Check write permissions with a test file
        try:
            test_file = os.path.join(dir_path, ".permission_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            logger.info(f"✓ {description} ({dir_path}) exists and is writable")
        except Exception as e:
            errors.append(f"Cannot write to {description} ({dir_path}): {e}")
    
    return errors, warnings

def _check_models() -> Tuple[List[str], List[str]]:
    """Check for required model files and validate their format if present"""
    errors = []
    warnings = []
    
    # Check for linear reward model
    reward_model_path = "models/final_model.pt"
    if os.path.exists(reward_model_path):
        try:
            # Try to load model to verify integrity
            _ = torch.load(reward_model_path, map_location="cpu")
            logger.info(f"✓ Linear reward model exists at {reward_model_path}")
        except Exception as e:
            errors.append(f"Linear reward model exists but is corrupt: {e}")
    else:
        warnings.append(f"Linear reward model not found at {reward_model_path}. Run train mode to create it.")
    
    # Check for PPO and NIM-PPO finetuned models (if they exist)
    model_dirs = {
        "models/ppo_finetuned": "PPO finetuned model",
        "models/nim_ppo_finetuned": "NIM-PPO finetuned model"
    }
    
    for model_dir, description in model_dirs.items():
        if os.path.exists(model_dir):
            # Check for minimum required files in a transformers model directory
            config_file = os.path.join(model_dir, "config.json")
            model_file = os.path.join(model_dir, "pytorch_model.bin")
            safetensors_exist = any(f.endswith('.safetensors') for f in os.listdir(model_dir)) if os.path.exists(model_dir) else False
            
            if os.path.exists(config_file) and (os.path.exists(model_file) or safetensors_exist):
                logger.info(f"✓ {description} exists at {model_dir}")
            else:
                warnings.append(f"{description} at {model_dir} appears incomplete (missing config.json or model weights)")
        else:
            # Models are optional, so just a warning
            warnings.append(f"{description} not found at {model_dir}")
    
    return errors, warnings

def check_huggingface_models(model_ids: List[str]) -> Tuple[List[str], List[str]]:
    """
    Check if the specified Hugging Face models are accessible.
    
    Args:
        model_ids: List of Hugging Face model IDs to check
        
    Returns:
        Tuple of (errors, warnings) lists
    """
    errors = []
    warnings = []
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        for model_id in model_ids:
            try:
                # Try to get model info
                model_info = api.model_info(model_id)
                logger.info(f"✓ Hugging Face model '{model_id}' is accessible")
                
                # Check if model requires acceptance of terms of use
                if model_info.gated:
                    warnings.append(f"Model '{model_id}' is gated and requires accepting terms of use on Hugging Face Hub")
                    
                # Check if model is private
                if not model_info.downloads:
                    warnings.append(f"Model '{model_id}' appears to be private or have restricted access")
                    
            except Exception as e:
                warnings.append(f"Could not access Hugging Face model '{model_id}': {e}")
    except ImportError:
        warnings.append("huggingface_hub package not installed. Cannot check model accessibility.")
    except Exception as e:
        warnings.append(f"Error checking Hugging Face models: {e}")
    
    return errors, warnings

def validate_model_file(model_path: str, name: str = None) -> Tuple[bool, str]:
    """
    Validate a model file or directory to ensure it exists and is in the correct format.
    
    Args:
        model_path: Path to the model file or directory
        name: Optional name for the model for better error messages
        
    Returns:
        (valid, error_message): Whether the model is valid and an error message if not
    """
    model_name = name or os.path.basename(model_path)
    
    if not os.path.exists(model_path):
        return False, f"{model_name} file not found at {model_path}"
    
    try:
        if model_path.endswith('.pt') or model_path.endswith('.bin'):
            # Try to load the PyTorch model
            _ = torch.load(model_path, map_location='cpu')
            return True, f"{model_name} file exists and is valid"
        else:
            # Check if directory contains expected files for transformers models
            config_file = os.path.join(model_path, "config.json")
            model_file = os.path.join(model_path, "pytorch_model.bin")
            safetensors_exist = any(f.endswith('.safetensors') for f in os.listdir(model_path)) if os.path.isdir(model_path) else False
            
            if not os.path.exists(config_file):
                return False, f"{model_name} may be incomplete: config.json not found in {model_path}"
                
            if not os.path.exists(model_file) and not safetensors_exist:
                return False, f"{model_name} may be incomplete: neither pytorch_model.bin nor safetensors found in {model_path}"
                
            return True, f"{model_name} directory exists and contains required files"
        
    except Exception as e:
        return False, f"Failed to validate {model_name} at {model_path}: {e}"

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run validation as standalone
    try:
        validate_environment()
        print("✅ All validation checks passed!")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)
