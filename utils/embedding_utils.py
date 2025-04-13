# utils/embedding_utils.py
import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
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
        
        # Configure Gemini API with the new client approach
        self.client = genai.Client(api_key=self.api_key)
        logger.info(f"Initialized Gemini Embedding with model: {model_id}")
    
    # utils/embedding_utils.py
    def get_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """Get embedding for text from Gemini Embedding model with fallback mechanisms"""
        retries = 0
        backoff_time = 1
        
        while retries < max_retries:
            try:
                # Primary approach
                embedding_model = self.client.get_model(self.model_id)
                content = types.Content(parts=[types.Part(text=text)])
                embedding_response = embedding_model.embed_content(
                    task_type=self.task_type,
                    content=content
                )
                
                # Extract embedding values
                if embedding_response and hasattr(embedding_response, "embedding"):
                    return embedding_response.embedding.values
                elif hasattr(embedding_response, "embeddings") and embedding_response.embeddings:
                    return embedding_response.embeddings[0].values
                    
            except Exception as primary_error:
                logger.warning(f"Primary embedding approach failed: {primary_error}")
                try:
                    # Fallback approach for different API versions
                    embeddings = self.client.embeddings.create(
                        model=self.model_id,
                        input=text
                    )
                    if hasattr(embeddings, "data") and len(embeddings.data) > 0:
                        return embeddings.data[0].embedding
                except Exception as fallback_error:
                    logger.warning(f"Fallback embedding approach also failed: {fallback_error}")
            
            retries += 1
            if retries < max_retries:
                sleep_time = backoff_time * (2 ** retries)
                logger.info(f"Retrying embedding in {sleep_time}s...")
                import time
                time.sleep(sleep_time)
        
        # Return empty embedding if all retries failed
        logger.warning("All embedding attempts failed, returning zeros")
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
