# starter.py
import os
import logging
import argparse
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start with a small test or run full implementation")
    parser.add_argument("--test", action="store_true", help="Run in test mode with small dataset")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Import validation
    from utils.validation import validate_environment
    logger.info("Validating environment...")
    validate_environment()
    
    if args.test:
        # Set environment variable for small dataset
        os.environ["USE_SMALL_DATASET"] = "True"
        os.environ["SMALL_DATASET_SIZE"] = "5"
        logger.info("Running in TEST mode with small dataset...")
    else:
        logger.info("Running in PRODUCTION mode with full dataset...")
    
    # Import and run main
    from main import main
    main()
