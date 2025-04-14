# REALM - A Novel Reward Model

This repository provides a production-ready implementation of a combined reward model that leverages both infly/INF-ORM-Llama3.1-70B for reward scoring and Google's Gemini embeddings for semantic similarity. The combined model can be used for Reinforcement Learning from Human Feedback (RLHF) to fine-tune large language models using techniques like PPO and DPO.

Architecture Overview

## Features

- 🔍 Combined reward signal using infly/INF-ORM-Llama3.1-70B and Gemini embeddings
- 🧠 Linear neural network trained on the Stanford Human Preferences (SHP) dataset
- 🚀 Optimized for memory efficiency with support for 4-bit and 8-bit quantization
- 📊 Robust caching, logging, and error handling for production deployment
- 🔄 Ready-to-use implementations for PPO and DPO fine-tuning
- 🐳 Docker support for reproducible deployment

## Models from Hugging Face for REALM
Reward Model - https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Google Gemini API key


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
export GEMINI_API_KEY=your_gemini_api_key
```

Alternatively, create a `.env` file:

```
GEMINI_API_KEY=your_gemini_api_key
```


## Project Structure

```
realm/
├── config/
│   └── config.yaml           # Configuration file
├── data/
│   └── processors.py         # Data processing utilities
├── models/
│   ├── llama_reward.py       # Llama 3.1 reward model wrapper
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
│   ├── logging_utils.py      # Logging utilities
│   └── embedding_utils.py    # Embedding utilities
├── main.py                   # Main entry point
├── api_server.py             # API server for inference
├── requirements.txt          # Dependencies
└── Dockerfile                # Docker configuration
```


## Usage

### Training the Combined Reward Model

Train the linear neural network to combine Llama 3.1 rewards and Gemini embeddings:

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


### Fine-tuning with DPO

Use the combined reward model for DPO fine-tuning:

```bash
python main.py --mode dpo --model_path models/final_model.pt --dataset_path data/prompts.json
```


## Configuration

The `config/config.yaml` file contains all configuration parameters:

```yaml
llama_reward:
  model_id: "infly/INF-ORM-Llama3.1-70B"
  quantization: "4bit"  # Options: "4bit", "8bit", or null for no quantization
  device_map: "auto"
  max_length: 2048

gemini:
  api_key: "${GEMINI_API_KEY}"
  model_id: "models/embedding-001"
  task_type: "retrieval_document"

data:
  dataset_name: "stanfordnlp/SHP"
  preprocessing:
    max_length: 1024
    batch_size: 16
    num_workers: 4
    cache_dir: "./cache"

model:
  input_dim: 2
  hidden_dims: [64, 32]
  output_dim: 1
  dropout: 0.1

training:
  seed: 42
  learning_rate: 1e-4
  weight_decay: 1e-5
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


## Docker Deployment

Build and run the Docker container:

```bash
docker build -t combined-reward-model .
docker run -e GEMINI_API_KEY=your_gemini_api_key combined-reward-model
```


## API Server

Start the API server for inference:

```bash
python api_server.py
```

Example API request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?", "response": "Machine learning is a field of AI that enables systems to learn from data."}'
```


## Model Architecture

The combined reward model uses a simple linear neural network to combine two features:

1. **Llama 3.1 Reward Score**: Generated by the infly/INF-ORM-Llama3.1-70B model, which evaluates the quality of the response.
2. **Semantic Similarity Score**: Calculated using cosine similarity between prompt and response Gemini embeddings.

The architecture consists of:

- Input layer (2 dimensions)
- Optional hidden layers
- Output layer (1 dimension) representing the final reward score


## Training Dataset

The model is trained on the [Stanford Human Preferences (SHP) dataset](https://huggingface.co/datasets/stanfordnlp/SHP), which contains:

- 385K human preferences over responses
- Coverage across 18 diverse domains
- Real-world preference data


## Performance Considerations

- **Memory Usage**: With 4-bit quantization, the Llama 3.1 70B model requires approximately 18GB of VRAM.
- **Throughput**: Batch processing is recommended for optimal performance.
- **API Costs**: Be mindful of Gemini API usage costs when processing large datasets.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [infly/INF-ORM-Llama3.1-70B](https://huggingface.co/infly/INF-ORM-Llama3.1-70B) for the reward model
- [Google Gemini API](https://ai.google.dev/gemini-api) for the embedding model
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

