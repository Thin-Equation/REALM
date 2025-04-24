# REALM - A Novel Reward Model

This repository provides a production-ready implementation of a combined reward model that leverages both NVIDIA NIM Llama 3.1 Nemotron 70B Reward model and Lajavaness embeddings for semantic similarity. The combined model can be used for Reinforcement Learning from Human Feedback (RLHF) to fine-tune large language models using techniques like PPO and DPO.

## Features

- 🔍 Combined reward signal using NVIDIA NIM Llama 3.1 Nemotron 70B Reward model and Lajavaness embeddings
- 🧠 Linear neural network trained on the Stanford Human Preferences (SHP) dataset
- 🚀 Optimized for memory efficiency with streamlined API interactions
- 📊 Robust caching, logging, and error handling for production deployment
- 🔄 Ready-to-use implementations for PPO and DPO fine-tuning
- 🐳 Docker support for reproducible deployment
- 📊 Evaluation framework for comparing reward model performance on the official TruthfulQA dataset
- ☁️ Training pipeline scripts for Brev and Lambda Labs platforms

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- NVIDIA NIM API key


### Setup

1. Clone the repository:
```bash
git clone https://github.com/Thin-Equation/realm.git
cd realm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export NVIDIA_NIM_API_KEY=your_nim_api_key
```

Alternatively, create a `.env` file:

```
NVIDIA_NIM_API_KEY=your_nim_api_key
```


## Project Structure

```
realm/
├── config/
│   └── config.yaml           # Configuration file
├── data/
│   └── processors.py         # Data processing utilities
├── models/
│   ├── nim_reward.py         # NIM Llama 3.1 Nemotron Reward model wrapper
│   └── reward_model.py       # Linear reward model implementation
├── training/
│   ├── trainer.py            # Training loop implementation
│   └── loss.py               # Loss functions
├── inference/
│   └── predictor.py          # Inference utilities
├── rlhf/
│   ├── ppo_integration.py    # PPO implementation
│   └── dpo_integration.py    # DPO implementation
├── utils/
│   ├── validation.py         # Environment validation utilities
│   └── embedding_utils.py    # Embedding utilities
├── main.py                   # Main entry point for all operations
├── brev_train_pipeline.sh    # Complete training pipeline script for Brev
├── lambda_train_pipeline.sh  # Complete training pipeline script for Lambda Labs
└── requirements.txt          # Dependencies
```


## Usage

### Training the Combined Reward Model

Train the linear neural network to combine NIM Llama 3.1 Nemotron rewards and Lajavaness embeddings:

```bash
python main.py --mode train --config config/config.yaml
```

### Running Inference

Get reward predictions for a prompt-response pair:

```bash
python main.py --mode predict --model_path models/final_model.pt --prompt "What is machine learning?" --response "Machine learning is a field of AI that enables systems to learn from data."
```


### Fine-tuning with PPO

Use the combined reward model for PPO fine-tuning:

```bash
python main.py --mode ppo --model_path models/final_model.pt --dataset_path data/prompts.json
```

Use only the NIM reward model for PPO fine-tuning (for comparison):

```bash
python main.py --mode nim_ppo --config config/config.yaml --output_dir models/nim_ppo_finetuned --max_steps 1000
```


### Fine-tuning with DPO

Use the combined reward model for DPO fine-tuning:

```bash
python main.py --mode dpo --model_path models/final_model.pt --dataset_path data/prompts.json
```


### Evaluating Models

Compare the performance of both fine-tuned models on the official TruthfulQA dataset:

```bash
python main.py --mode evaluate --combined_model_path models/ppo_finetuned --nim_model_path models/nim_ppo_finetuned --output_dir evaluation_results
```


## Configuration

The `config/config.yaml` file contains all configuration parameters:

```yaml
nim_reward:
  api_key: "${NVIDIA_NIM_API_KEY}"
  base_url: "https://integrate.api.nvidia.com/v1"
  model_id: "nvidia/llama-3.1-nemotron-70b-reward"
  max_retries: 3
  retry_delay: 2.0

embedding:
  model_id: "Lajavaness/bilingual-embedding-large"

data:
  dataset_name: "stanfordnlp/SHP"
  preprocessing:
    max_length: 1024
    batch_size: 16
    num_workers: 0
    cache_dir: "./cache"

model:
  input_dim: 2
  hidden_dims: [64, 32]
  output_dim: 1
  dropout: 0.1

training:
  seed: 42
  learning_rate: 0.0001
  weight_decay: 0.01
  num_epochs: 10
  # Additional training parameters...

rlhf:
  ppo:
    model_name: "meta-llama/Llama-3.1-8B-Instruct"
    batch_size: 8
    # Additional PPO parameters...
  
  dpo:
    model_name: "meta-llama/Llama-3.1-8B-Instruct"
    learning_rate: 5e-7
    # Additional DPO parameters...
```

## Model Architecture

The combined reward model uses a simple linear neural network to combine two features:

1. **NIM Llama 3.1 Nemotron Reward Score**: Generated by the NVIDIA NIM reward model API, which evaluates the quality of the response.
2. **Semantic Similarity Score**: Calculated using cosine similarity between prompt and response Lajavaness embeddings.

The architecture consists of:

- Input layer (2 dimensions)
- Hidden layers [64, 32]
- Output layer (1 dimension) representing the final reward score


## Training Dataset

The model is trained on the [Stanford Human Preferences (SHP) dataset](https://huggingface.co/datasets/stanfordnlp/SHP), which contains:

- 385K human preferences over responses
- Coverage across 18 diverse domains
- Real-world preference data


## Evaluation Dataset

For evaluating model truthfulness, we use the official [TruthfulQA dataset](https://huggingface.co/datasets/truthful_qa) from Hugging Face with the "multiple_choice" configuration. This dataset is designed to measure a model's ability to avoid generating false or misleading information.


## Performance Considerations

- **API Usage**: The NIM Reward model is accessed via API, so be mindful of usage limits and costs.
- **Throughput**: Batch processing and caching are implemented for optimal performance.
- **GPU Memory**: Training and inference have been optimized to run on consumer GPUs while still leveraging state-of-the-art models.


## Complete Training Pipeline

### Running on Brev

To run the complete training pipeline on Brev:

```bash
./brev_train_pipeline.sh
```

This script will:
1. Train the Linear Reward Model
2. Fine-tune a model using the combined reward with PPO
3. Fine-tune a model using only the NIM reward with PPO
4. Evaluate both fine-tuned models on the TruthfulQA dataset
5. Verify all model weights were saved correctly

After completion, you can download the results from Brev using:
```bash
brev pull <instance-name>:/app/models/final_model.pt ./models/
brev pull -r <instance-name>:/app/models/ppo_finetuned ./models/
brev pull -r <instance-name>:/app/models/nim_ppo_finetuned ./models/
brev pull -r <instance-name>:/app/evaluation_results ./evaluation_results/
```

### Running on Lambda Labs

To run the complete training pipeline on Lambda Labs:

```bash
./lambda_train_pipeline.sh
```

The Lambda Labs script includes additional features:
- Comprehensive system and GPU information reporting
- Automatic environment detection (conda vs pip)
- Optional Weights & Biases integration (`export WANDB_API_KEY=your_key`)
- Automatic result compression for easy download

After completion, all results are compressed into `realm_results.tar.gz` which you can download from Lambda Labs using:
```bash
scp <username>@<lambda-instance-ip>:/path/to/realm_results.tar.gz /local/path/
```
Or through the Lambda Labs web console.


## Verifying Model Weights

To verify that all model weights are properly saved:

```bash
python main.py --mode verify
```

This will check for the existence of:
1. Reward Model (LinearRewardModel): `models/final_model.pt`
2. Combined-Reward Fine-tuned LLM: `models/ppo_finetuned/`
3. NIM-Reward Fine-tuned LLM: `models/nim_ppo_finetuned/`

## Acknowledgements

- [NVIDIA NIM Llama-3.1-Nemotron-70B-Reward](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward) for the reward model 
- [Lajavaness/bilingual-embedding-large](https://huggingface.co/Lajavaness/bilingual-embedding-large) for the embedding model
- [Stanford NLP](https://huggingface.co/datasets/stanfordnlp/SHP) for the SHP dataset
- [TruthfulQA](https://huggingface.co/datasets/truthful_qa) for the evaluation dataset
- [TRL library](https://github.com/huggingface/trl) for RLHF implementation


## Citation

If you use this code in your research, please cite:

```bibtex
@software{realm,
  author = {Dhairya Gundechia, Suhas KM, William Claster},
  title = {REALM},
  year = {2025},
  url = {https://github.com/Thin-Equation/realm},
}
```

