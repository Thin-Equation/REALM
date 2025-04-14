# utils/embedding_utils.py
import logging
import numpy as np
from typing import List
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
        """Get embedding for text from Gemini Embedding model"""
        retries = 0
        backoff_time = 1
        
        while retries < max_retries:
            try:
                # The correct method for google-genai >= 1.10.0
                embedding_result = self.client.models.embed_content(
                    model=self.model_id,
                    contents=text,
                    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
                )
                
                # Access the values correctly
                if hasattr(embedding_result, "embeddings") and embedding_result.embeddings:
                    # Extract the values from the first embedding
                    return embedding_result.embeddings[0].values
                
                # Alternative format that might be used
                if hasattr(embedding_result, "embedding"):
                    return embedding_result.embedding.values
                
                logger.warning(f"Unexpected embedding result format: {embedding_result}")
                
            except Exception as e:
                logger.warning(f"Embedding attempt {retries+1} failed: {e}")
                
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
