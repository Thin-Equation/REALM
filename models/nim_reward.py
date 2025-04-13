# models/nim_reward.py
import os
import time
import logging
import requests
import threading
from typing import Dict, List, Union, Optional, Any
import torch
import numpy as np
from queue import Queue, Empty

logger = logging.getLogger(__name__)

class RateLimiter:
    """Thread-safe rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_period: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the time period
            time_period: Time period in seconds (default: 60 seconds = 1 minute)
        """
        self.max_calls = max_calls
        self.time_period = time_period
        self.calls = []
        self.lock = threading.Lock()
    
    def _cleanup_old_calls(self):
        """Remove calls that are outside the time period"""
        current_time = time.time()
        with self.lock:
            self.calls = [call_time for call_time in self.calls 
                         if current_time - call_time < self.time_period]
    
    def wait(self):
        """Wait until a call can be made"""
        while True:
            self._cleanup_old_calls()
            with self.lock:
                if len(self.calls) < self.max_calls:
                    # We can make a call
                    self.calls.append(time.time())
                    return
            
            # Calculate the optimal wait time based on the oldest call
            with self.lock:
                if self.calls:
                    oldest_call = min(self.calls)
                    wait_time = max(0.1, self.time_period - (time.time() - oldest_call))
                else:
                    wait_time = 0.1
            
            # Wait before checking again
            time.sleep(wait_time)
    
    def __call__(self, func):
        """Decorator for rate-limited functions"""
        def wrapper(*args, **kwargs):
            self.wait()
            return func(*args, **kwargs)
        return wrapper


class NimLlamaRewardModel:
    """Wrapper for llama-3.1-nemotron-70b-reward model via Nvidia NIM API"""
    
    def __init__(
        self, 
        api_key: str,
        api_url: str = "https://api.nim.nvidia.com/v1/llm",
        model_id: str = "llama-3.1-nemotron-70b-reward",
        max_calls_per_minute: int = 40,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        max_batch_size: int = 10,
    ):
        """
        Initialize the Llama 3.1 Nemotron 70B Reward model via NIM API.
        
        Args:
            api_key: Nvidia NIM API key
            api_url: Nvidia NIM API endpoint URL
            model_id: Model ID to use for inference
            max_calls_per_minute: Maximum API calls per minute (rate limit)
            max_retries: Maximum number of retries for failed API calls
            retry_delay: Initial delay between retries (will be exponentially increased)
            max_batch_size: Maximum batch size for batch inference
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model_id = model_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_batch_size = max_batch_size
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(max_calls_per_minute)
        
        # Initialize request session
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        logger.info(f"Initialized NIM Llama Reward Model: {model_id}")
    
    def format_prompt(self, prompt: str, response: str) -> str:
        """Format prompt and response for the reward model"""
        # Format for Llama 3.1 Nemotron reward model
        formatted_text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}"
        return formatted_text
    
    def _call_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a rate-limited call to the NIM API
        
        Args:
            payload: Request payload to send to the API
            
        Returns:
            API response as a dictionary
        """
        # Apply rate limiting
        self.rate_limiter.wait()
        
        url = f"{self.api_url}/completions"
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(url, json=payload)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit exceeded
                    retry_after = int(response.headers.get('Retry-After', self.retry_delay * (2 ** attempt)))
                    logger.warning(f"Rate limit exceeded. Waiting for {retry_after} seconds.")
                    time.sleep(retry_after)
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    if attempt < self.max_retries - 1:
                        sleep_time = self.retry_delay * (2 ** attempt)
                        logger.info(f"Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                    else:
                        raise Exception(f"API error: {response.status_code} - {response.text}")
            
            except Exception as e:
                logger.error(f"API call failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    raise e
        
        raise Exception(f"Failed to call API after {self.max_retries} attempts")
    
    def get_reward_score(self, prompt: str, response: str) -> float:
        """Get reward score for a prompt-response pair"""
        try:
            # Format the input
            formatted_input = self.format_prompt(prompt, response)
            
            # Prepare the payload for Nemotron reward model
            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ],
                "temperature": 0.0,  # We want deterministic scoring
                "task_type": "reward",  # Specify the task type for reward models
            }
            
            # Call the API with rate limiting
            result = self._call_api(payload)
            
            # Extract reward score from the response
            # Adjust based on the actual NIM API response format
            if "choices" in result and len(result["choices"]) > 0:
                if "reward" in result["choices"][0]:
                    reward = result["choices"][0]["reward"]
                else:
                    # Extract from metadata if provided in a different format
                    reward = result.get("reward_score", 0.0)
            else:
                logger.warning(f"Unexpected API response format: {result}")
                reward = 0.0
            
            return float(reward)
        
        except Exception as e:
            logger.error(f"Error getting reward score: {str(e)}")
            # Return default score in case of error
            return 0.0
    
    def batch_get_reward_scores(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Get reward scores for batches of prompt-response pairs"""
        if len(prompts) != len(responses):
            raise ValueError("Number of prompts and responses must be the same")
        
        all_rewards = []
        
        # Process in smaller batches to respect API limitations
        for i in range(0, len(prompts), self.max_batch_size):
            batch_prompts = prompts[i:i+self.max_batch_size]
            batch_responses = responses[i:i+self.max_batch_size]
            
            batch_rewards = []
            for prompt, response in zip(batch_prompts, batch_responses):
                reward = self.get_reward_score(prompt, response)
                batch_rewards.append(reward)
            
            all_rewards.extend(batch_rewards)
        
        return all_rewards


class BatchProcessingNimLlamaRewardModel(NimLlamaRewardModel):
    """
    Enhanced version of NimLlamaRewardModel with parallel processing using a worker pool
    for efficient handling of rate limits
    """
    
    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.nim.nvidia.com/v1/llm",
        model_id: str = "llama-3.1-nemotron-70b-reward",
        max_calls_per_minute: int = 40,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        max_batch_size: int = 10,
        num_workers: int = 4,
    ):
        """
        Initialize with worker pool for parallel processing within rate limits
        
        Args:
            num_workers: Number of worker threads for parallel processing
            (other args same as parent class)
        """
        super().__init__(
            api_key=api_key,
            api_url=api_url,
            model_id=model_id,
            max_calls_per_minute=max_calls_per_minute,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_batch_size=max_batch_size,
        )
        
        self.num_workers = num_workers
        
        # Initialize work queue and worker threads
        self.queue = Queue()
        self.result_queue = Queue()
        self.workers = []
        
        # Start worker threads
        for _ in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self):
        """Worker thread that processes reward requests from the queue"""
        while True:
            try:
                # Get a task from the queue
                task_id, prompt, response = self.queue.get()
                
                # Process the task
                try:
                    reward = super().get_reward_score(prompt, response)
                    self.result_queue.put((task_id, reward))
                except Exception as e:
                    logger.error(f"Error in worker thread: {str(e)}")
                    self.result_queue.put((task_id, 0.0))
                
                # Mark the task as done
                self.queue.task_done()
            
            except Empty:
                # Queue is empty, wait a bit
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Unexpected error in worker thread: {str(e)}")
                time.sleep(1.0)
    
    def batch_get_reward_scores(self, prompts: List[str], responses: List[str]) -> List[float]:
        """
        Get reward scores for batches of prompt-response pairs using worker pool.
        This method enables parallel processing within the rate limits.
        """
        if len(prompts) != len(responses):
            raise ValueError("Number of prompts and responses must be the same")
        
        # Add all tasks to the queue
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            self.queue.put((i, prompt, response))
        
        # Wait for all tasks to be processed
        self.queue.join()
        
        # Collect results
        results = {}
        while not self.result_queue.empty():
            task_id, reward = self.result_queue.get()
            results[task_id] = reward
        
        # Organize results in the original order
        return [results.get(i, 0.0) for i in range(len(prompts))]
