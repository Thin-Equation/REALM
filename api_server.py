# api_server.py
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import torch
import yaml
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional

from utils.logging_utils import setup_logging
from models.nim_reward import BatchProcessingNimLlamaRewardModel
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

# Load Llama 3.1 Nemotron Reward model via NIM API
logger.info("Initializing NIM Llama Reward model...")
nim_reward_model = BatchProcessingNimLlamaRewardModel(
    api_key=config["nim_reward"]["api_key"],
    api_url=config["nim_reward"]["api_url"],
    model_id=config["nim_reward"]["model_id"],
    max_calls_per_minute=config["nim_reward"]["max_calls_per_minute"],
    max_retries=config["nim_reward"]["max_retries"],
    retry_delay=config["nim_reward"]["retry_delay"],
    max_batch_size=config["nim_reward"]["max_batch_size"],
    num_workers=config["nim_reward"]["num_workers"]
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
    nim_reward_model=nim_reward_model,
    gemini_embedding=gemini_embedding,
    device=device
)

# Create FastAPI app
app = FastAPI(
    title="Combined Reward Model API",
    description="API for combined Llama 3.1 Nemotron 70B Reward and Gemini Embedding reward model",
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

# Rate limiter for API endpoints
class APIRateLimiter:
    """Rate limiter for API endpoints"""
    
    def __init__(self, max_calls: int, time_period: int = 60):
        self.max_calls = max_calls
        self.time_period = time_period  # in seconds
        self.calls = []
        self.lock = asyncio.Lock()
    
    async def _cleanup_old_calls(self):
        """Remove calls that are outside the time period"""
        current_time = time.time()
        async with self.lock:
            self.calls = [call_time for call_time in self.calls 
                         if current_time - call_time < self.time_period]
    
    async def is_rate_limited(self) -> bool:
        """Check if current request exceeds rate limit"""
        await self._cleanup_old_calls()
        async with self.lock:
            if len(self.calls) >= self.max_calls:
                return True
            
            # Add current call
            self.calls.append(time.time())
            return False

# Create rate limiter instance - 100 calls per minute to the API endpoints
api_rate_limiter = APIRateLimiter(max_calls=100, time_period=60)

# Dependency for rate limiting
async def check_rate_limit():
    if await api_rate_limiter.is_rate_limited():
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Please try again later."
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

# Queue for batch processing
batch_queue = asyncio.Queue()
batch_results = {}

# Background batch processing task
async def batch_processor():
    """Process batches of requests from the queue"""
    while True:
        try:
            batch_id, prompts, responses = await batch_queue.get()
            
            # Process the batch
            try:
                rewards = predictor.batch_predict(prompts, responses)
                batch_results[batch_id] = {
                    "status": "completed",
                    "rewards": rewards
                }
            except Exception as e:
                logger.error(f"Error processing batch {batch_id}: {str(e)}")
                batch_results[batch_id] = {
                    "status": "failed",
                    "error": str(e)
                }
            
            # Clean up old results after some time
            batch_queue.task_done()
            
        except Exception as e:
            logger.error(f"Error in batch processor: {str(e)}")
            await asyncio.sleep(1)

# Start the batch processor on app startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_processor())

# Define routes
@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(check_rate_limit)])
async def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    """Predict reward for a prompt-response pair"""
    try:
        # Get Llama score (this will respect the rate limiter inside the model)
        llama_score = nim_reward_model.get_reward_score(request.prompt, request.response)
        
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

@app.post("/batch_predict", dependencies=[Depends(check_rate_limit)])
async def batch_predict(request: BatchPredictRequest):
    """Start an asynchronous batch prediction task"""
    try:
        if len(request.prompts) != len(request.responses):
            raise ValueError("Number of prompts and responses must be the same")
        
        # Create a unique batch ID
        batch_id = f"batch_{int(time.time())}_{hash(tuple(request.prompts))}"
        
        # Add to the queue for background processing
        await batch_queue.put((batch_id, request.prompts, request.responses))
        
        # Return the batch ID for later status checking
        return {"batch_id": batch_id, "status": "submitted", "count": len(request.prompts)}
    
    except Exception as e:
        logger.error(f"Error submitting batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/batch_status/{batch_id}")
async def batch_status(batch_id: str):
    """Check the status of a batch prediction task"""
    if batch_id not in batch_results:
        return {"batch_id": batch_id, "status": "not_found"}
    
    return {"batch_id": batch_id, **batch_results[batch_id]}

@app.post("/compare", response_model=CompareResponse, dependencies=[Depends(check_rate_limit)])
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
