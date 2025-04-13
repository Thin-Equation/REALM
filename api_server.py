# api_server.py
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import torch
import yaml
import logging
from typing import List, Dict, Any, Optional

from utils.logging_utils import setup_logging
from models.llama_reward import LlamaRewardModel
from utils.embedding_utils import GeminiEmbedding
from models.reward_model import LinearRewardModel
from inference.predictor import RewardPredictor

# Setup logging
logger = setup_logging()

# Load configuration
def load_config():
    config_path = os.environ.get("CONFIG_PATH", "config/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Replace environment variables in config
    for section in config:
        if isinstance(config[section], dict):
            for key, value in config[section].items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    config[section][key] = os.environ.get(env_var, "")
    
    return config

config = load_config()

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Llama 3.1 Reward model
logger.info("Loading Llama 3.1 Reward model...")
llama_reward_model = LlamaRewardModel(
    model_id=config["llama_reward"]["model_id"],
    quantization=config["llama_reward"]["quantization"],
    device_map=config["llama_reward"]["device_map"],
    max_length=config["llama_reward"]["max_length"]
)

# Initialize Gemini Embedding
logger.info("Initializing Gemini Embedding...")
gemini_embedding = GeminiEmbedding(
    api_key=config["gemini"]["api_key"],
    model_id=config["gemini"]["model_id"],
    task_type=config["gemini"]["task_type"]
)

# Load combined reward model
model_path = os.environ.get("MODEL_PATH", "models/final_model.pt")
logger.info(f"Loading combined reward model from {model_path}...")
predictor = RewardPredictor(
    model_path=model_path,
    llama_reward_model=llama_reward_model,
    gemini_embedding=gemini_embedding,
    device=device
)

# Create FastAPI app
app = FastAPI(
    title="Combined Reward Model API",
    description="API for combined Llama 3.1 70B Reward and Gemini Embedding reward model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request/response models
class PredictRequest(BaseModel):
    prompt: str
    response: str

class PredictResponse(BaseModel):
    reward: float
    llama_score: Optional[float]
    similarity_score: Optional[float]

class BatchPredictRequest(BaseModel):
    prompts: List[str]
    responses: List[str]

class BatchPredictResponse(BaseModel):
    rewards: List[float]

class CompareRequest(BaseModel):
    prompt: str
    response1: str
    response2: str

class CompareResponse(BaseModel):
    reward1: float
    reward2: float
    better_response: int  # 1 or 2

# Define routes
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict reward for a prompt-response pair"""
    try:
        # Get Llama score
        llama_score = llama_reward_model.get_reward_score(request.prompt, request.response)
        
        # Get similarity score
        prompt_embedding = gemini_embedding.get_embedding(request.prompt)
        response_embedding = gemini_embedding.get_embedding(request.response)
        from utils.embedding_utils import cosine_similarity
        similarity_score = cosine_similarity(prompt_embedding, response_embedding)
        
        # Get combined reward
        reward = predictor.predict(request.prompt, request.response)
        
        return PredictResponse(
            reward=reward,
            llama_score=llama_score,
            similarity_score=similarity_score
        )
    except Exception as e:
        logger.error(f"Error predicting reward: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict", response_model=BatchPredictResponse)
async def batch_predict(request: BatchPredictRequest):
    """Predict rewards for a batch of prompt-response pairs"""
    try:
        if len(request.prompts) != len(request.responses):
            raise ValueError("Number of prompts and responses must be the same")
        
        rewards = predictor.batch_predict(request.prompts, request.responses)
        return BatchPredictResponse(rewards=rewards)
    except Exception as e:
        logger.error(f"Error batch predicting rewards: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest):
    """Compare two responses for the same prompt"""
    try:
        reward1, reward2, better = predictor.compare(request.prompt, request.response1, request.response2)
        return CompareResponse(
            reward1=reward1,
            reward2=reward2,
            better_response=better
        )
    except Exception as e:
        logger.error(f"Error comparing responses: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
