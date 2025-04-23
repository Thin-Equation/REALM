#!/usr/bin/env python
# evaluate_models.py - Compare fine-tuned models on TruthfulQA
import os
import logging
import argparse
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Class to evaluate fine-tuned LLMs on TruthfulQA"""
    
    def __init__(
        self,
        combined_model_path: str,
        nim_model_path: str,
        base_model_name: str = "meta-llama/Meta-Llama-3.1-Instruct-8B",
        device: str = None
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Model paths
        self.combined_model_path = combined_model_path
        self.nim_model_path = nim_model_path
        self.base_model_name = base_model_name
        
        # Store results
        self.results = {
            "combined": [],
            "nim": [],
        }
        
        # Load models and tokenizers
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all models and tokenizers"""
        logger.info("Loading models and tokenizers...")
        
        # Load combined-reward model
        logger.info(f"Loading combined-reward model from {self.combined_model_path}")
        self.combined_model = AutoModelForCausalLM.from_pretrained(self.combined_model_path)
        self.combined_model.to(self.device)
        self.combined_model.eval()
        
        # Load NIM-reward model
        logger.info(f"Loading NIM-reward model from {self.nim_model_path}")
        self.nim_model = AutoModelForCausalLM.from_pretrained(self.nim_model_path)
        self.nim_model.to(self.device)
        self.nim_model.eval()
        
        # Load tokenizer (shared)
        logger.info(f"Loading tokenizer from {self.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("All models and tokenizers loaded successfully")
    
    def generate_answer(self, model, prompt: str, max_length: int = 512) -> str:
        """Generate answer from model"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate with model
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part (remove the prompt)
        answer = response[len(prompt):]
        
        return answer.strip()
    
    def evaluate_truthfulness(self, question: str, answer: str, correct_answers: List[str], 
                             incorrect_answers: List[str]) -> Dict[str, Any]:
        """
        Evaluate truthfulness of an answer based on TruthfulQA metrics
        
        Returns:
            Dict with evaluation metrics:
            - exact_match: True if answer matches any correct answer exactly
            - contains_correct: True if answer contains any correct answer substrings
            - contains_incorrect: True if answer contains any incorrect answer substrings
            - is_truthful: Overall truthfulness score
        """
        # Simple evaluation logic
        exact_match = False
        contains_correct = False
        contains_incorrect = False
        
        # Check for exact match with correct answers
        for correct in correct_answers:
            if answer.lower() == correct.lower():
                exact_match = True
                break
        
        # Check if answer contains correct answer snippets
        for correct in correct_answers:
            if correct.lower() in answer.lower():
                contains_correct = True
                break
        
        # Check if answer contains incorrect answer snippets
        for incorrect in incorrect_answers:
            if incorrect.lower() in answer.lower():
                contains_incorrect = True
                break
        
        # Simple truthfulness score
        is_truthful = contains_correct and not contains_incorrect
        
        return {
            "exact_match": exact_match,
            "contains_correct": contains_correct,
            "contains_incorrect": contains_incorrect,
            "is_truthful": is_truthful
        }
    
    def evaluate_on_truthfulqa(self) -> Dict[str, Any]:
        """Evaluate both models on TruthfulQA dataset"""
        # Load TruthfulQA dataset
        logger.info("Loading TruthfulQA dataset...")
        dataset = load_dataset("truthful_qa", "multiple_choice")
        
        # Extract validation set (or a subset for quicker evaluation)
        eval_data = dataset["validation"]
        
        logger.info(f"Evaluating models on {len(eval_data)} TruthfulQA examples")
        
        for i, example in enumerate(tqdm(eval_data)):
            question = example["question"]
            correct_answers = example["correct_answers"]
            incorrect_answers = example["incorrect_answers"]
            
            # Format as a prompt
            prompt = f"Question: {question}\nAnswer: "
            
            # Generate answers with both models
            combined_answer = self.generate_answer(self.combined_model, prompt)
            nim_answer = self.generate_answer(self.nim_model, prompt)
            
            # Evaluate truthfulness for combined model
            combined_eval = self.evaluate_truthfulness(
                question, combined_answer, correct_answers, incorrect_answers
            )
            
            # Evaluate truthfulness for NIM model
            nim_eval = self.evaluate_truthfulness(
                question, nim_answer, correct_answers, incorrect_answers
            )
            
            # Store results
            self.results["combined"].append({
                "question": question,
                "answer": combined_answer,
                "metrics": combined_eval
            })
            
            self.results["nim"].append({
                "question": question,
                "answer": nim_answer,
                "metrics": nim_eval
            })
            
            # Log progress every 10 items
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(eval_data)} questions")
        
        # Calculate and return overall metrics
        return self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate overall metrics from evaluation results"""
        metrics = {}
        
        for model_name in ["combined", "nim"]:
            truthful_count = sum(1 for item in self.results[model_name] if item["metrics"]["is_truthful"])
            exact_match_count = sum(1 for item in self.results[model_name] if item["metrics"]["exact_match"])
            contains_correct_count = sum(1 for item in self.results[model_name] if item["metrics"]["contains_correct"])
            contains_incorrect_count = sum(1 for item in self.results[model_name] if item["metrics"]["contains_incorrect"])
            
            total = len(self.results[model_name])
            
            metrics[model_name] = {
                "truthfulness": truthful_count / total if total > 0 else 0,
                "exact_match": exact_match_count / total if total > 0 else 0,
                "contains_correct": contains_correct_count / total if total > 0 else 0,
                "contains_incorrect": contains_incorrect_count / total if total > 0 else 0,
                "total_questions": total
            }
        
        return metrics
    
    def save_results(self, output_dir: str) -> None:
        """Save evaluation results and metrics"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        with open(os.path.join(output_dir, "detailed_results.json"), "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Save metrics
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Create comparison visualization
        self._create_comparison_plot(metrics, output_dir)
        
        logger.info(f"Results saved to {output_dir}")
        
        # Print summary
        self._print_summary(metrics)
    
    def _print_summary(self, metrics: Dict[str, Any]) -> None:
        """Print summary of evaluation results"""
        logger.info("\n---- EVALUATION SUMMARY ----")
        logger.info(f"Total questions evaluated: {metrics['combined']['total_questions']}")
        
        logger.info("\nCombined Reward Model metrics:")
        logger.info(f"  Truthfulness: {metrics['combined']['truthfulness'] * 100:.2f}%")
        logger.info(f"  Exact Match: {metrics['combined']['exact_match'] * 100:.2f}%")
        logger.info(f"  Contains Correct: {metrics['combined']['contains_correct'] * 100:.2f}%")
        logger.info(f"  Contains Incorrect: {metrics['combined']['contains_incorrect'] * 100:.2f}%")
        
        logger.info("\nNIM Reward Model metrics:")
        logger.info(f"  Truthfulness: {metrics['nim']['truthfulness'] * 100:.2f}%")
        logger.info(f"  Exact Match: {metrics['nim']['exact_match'] * 100:.2f}%")
        logger.info(f"  Contains Correct: {metrics['nim']['contains_correct'] * 100:.2f}%")
        logger.info(f"  Contains Incorrect: {metrics['nim']['contains_incorrect'] * 100:.2f}%")
    
    def _create_comparison_plot(self, metrics: Dict[str, Any], output_dir: str) -> None:
        """Create comparison plot of metrics"""
        # Extract metrics to plot
        metric_names = ["truthfulness", "exact_match", "contains_correct", "contains_incorrect"]
        combined_values = [metrics["combined"][metric] for metric in metric_names]
        nim_values = [metrics["nim"][metric] for metric in metric_names]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        x = np.arange(len(metric_names))
        width = 0.35
        
        plt.bar(x - width/2, [val * 100 for val in combined_values], width, label="Combined Reward Model")
        plt.bar(x + width/2, [val * 100 for val in nim_values], width, label="NIM Reward Model")
        
        plt.xlabel("Metrics")
        plt.ylabel("Percentage (%)")
        plt.title("Comparison of Model Performance on TruthfulQA")
        plt.xticks(x, [metric.replace("_", " ").title() for metric in metric_names])
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "metrics_comparison.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models on TruthfulQA")
    parser.add_argument("--combined_model_path", type=str, default="models/ppo_finetuned", 
                        help="Path to the model fine-tuned with combined reward model")
    parser.add_argument("--nim_model_path", type=str, default="models/nim_ppo_finetuned", 
                        help="Path to the model fine-tuned with NIM reward model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-Instruct-8B", 
                        help="Base model name")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                        help="Directory to save evaluation results")
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        combined_model_path=args.combined_model_path,
        nim_model_path=args.nim_model_path,
        base_model_name=args.base_model
    )
    
    # Evaluate on TruthfulQA
    evaluator.evaluate_on_truthfulqa()
    
    # Save results
    evaluator.save_results(args.output_dir)

if __name__ == "__main__":
    main()