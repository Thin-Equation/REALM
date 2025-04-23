# REALM - A Novel Reward Model

This repository provides a production-ready implementation of a combined reward model that leverages both NVIDIA NIM Llama 3.1 Nemotron 70B Reward model and Lajavaness embeddings for semantic similarity. The combined model can be used for Reinforcement Learning from Human Feedback (RLHF) to fine-tune large language models using techniques like PPO and DPO.

Architecture Overview

## Features

- 🔍 Combined reward signal using NVIDIA NIM Llama 3.1 Nemotron 70B Reward model and Lajavaness embeddings
- 🧠 Linear neural network trained on the Stanford Human Preferences (SHP) dataset
- 🚀 Optimized for memory efficiency with streamlined API interactions
- 📊 Robust caching, logging, and error handling for production deployment
- 🔄 Ready-to-use implementations for PPO and DPO fine-tuning
- 🐳 Docker support for reproducible deployment
- 📊 Evaluation framework for comparing reward model performance on TruthfulQA

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
├── main.py                   # Main entry point
├── nim_ppo_finetune.py       # NIM-only PPO fine-tuning script
├── evaluate_models.py        # Model evaluation script
├── api_server.py             # API server for inference
└── requirements.txt          # Dependencies
```


## Usage

### Training the Combined Reward Model

Train the linear neural network to combine NIM Llama 3.1 Nemotron rewards and Lajavaness embeddings:

```bash
python main.py --mode train --config config/config.yaml
```


### Evaluating the Model

Evaluate the trained model on the test set:

```bash
python main.py --mode eval --model_path models/final_model.pt --config config/config.yaml
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
python nim_ppo_finetune.py --config config/config.yaml --output_dir models/nim_ppo_finetuned --max_steps 1000
```


### Fine-tuning with DPO

Use the combined reward model for DPO fine-tuning:

```bash
python main.py --mode dpo --model_path models/final_model.pt --dataset_path data/prompts.json
```


### Comparing Models

Compare the performance of both fine-tuned models on the TruthfulQA dataset:

```bash
python evaluate_models.py --combined_model_path models/ppo_finetuned --nim_model_path models/nim_ppo_finetuned --output_dir evaluation_results
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
    model_name: "meta-llama/Meta-Llama-3.1-Instruct-8B"
    batch_size: 8
    # Additional PPO parameters...
  
  dpo:
    model_name: "meta-llama/Meta-Llama-3.1-Instruct-8B"
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


## Performance Considerations

- **API Usage**: The NIM Reward model is accessed via API, so be mindful of usage limits and costs.
- **Throughput**: Batch processing and caching are implemented for optimal performance.
- **Training on Brev**: For training on NVIDIA Brev, use the provided `brev_train_pipeline.sh` script.


## Verifying Model Weights

To verify that all model weights are properly saved:

```bash
python verify_models.py
```

This will check for the existence of:
1. Reward Model (LinearRewardModel): `models/final_model.pt`
2. Combined-Reward Fine-tuned LLM: `models/ppo_finetuned/`
3. NIM-Reward Fine-tuned LLM: `models/nim_ppo_finetuned/`

## Acknowledgements

- [NVIDIA NIM Llama-3.1-Nemotron-70B-Reward](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward) for the reward model 
- [Lajavaness/bilingual-embedding-large](https://huggingface.co/Lajavaness/bilingual-embedding-large) for the embedding model
- [Stanford NLP](https://huggingface.co/datasets/stanfordnlp/SHP) for the SHP dataset
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

