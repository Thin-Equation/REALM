# utils/embedding_utils.py
import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class GeminiEmbedding:
    """Wrapper for the Google Gemini API to get embeddings"""
    
    def __init__(self, api_key: str, model_id: str, task_type: str = "retrieval_document"):
        """
        Initialize the Gemini Embedding API client.
        
        Args:
            api_key: Gemini API key
            model_id: Gemini embedding model ID
            task_type: Task type for embedding generation
        """
        self.api_key = api_key
        self.model_id = model_id
        self.task_type = task_type
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        logger.info(f"Initialized Gemini Embedding with model: {model_id}")
    
    def get_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """Get embedding for text from Gemini Embedding model"""
        retries = 0
        backoff_time = 1
        
        while retries < max_retries:
            try:
                embedding = genai.embed_content(
                    model=self.model_id,
                    content=text,
                    task_type=self.task_type
                )
                
                # Extract embedding values
                if embedding and hasattr(embedding, "embedding"):
                    return embedding.embedding
                elif embedding and isinstance(embedding, dict) and "embedding" in embedding:
                    return embedding["embedding"]
                elif embedding and isinstance(embedding, dict) and "embeddings" in embedding:
                    return embedding["embeddings"][0]["values"]
                else:
                    logger.error(f"Unexpected Gemini embedding response format: {embedding}")
                    retries += 1
                    backoff_time *= 2
                    import time
                    time.sleep(backoff_time)
            
            except Exception as e:
                logger.error(f"Exception calling Gemini API: {str(e)}")
                retries += 1
                backoff_time *= 2
                import time
                time.sleep(backoff_time)
        
        # Return empty embedding if all retries failed
        logger.warning("All retries failed for Gemini API call, returning empty embedding")
        return [0.0] * 768  # Default dimension for Gemini embeddings
    
    def batch_get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts"""
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if not vec1 or not vec2:
        return 0.0
    
    # Convert to numpy arrays
    v1 = np.array(vec1).reshape(1, -1)
    v2 = np.array(vec2).reshape(1, -1)
    
    # Calculate cosine similarity
    similarity = sklearn_cosine_similarity(v1, v2)[0][0]
    return float(similarity)
