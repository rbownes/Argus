"""
Implementation of the TruthfulQA benchmark.
"""
import json
import os
import random
import re
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Union
import aiofiles

from llm_eval.core.models import Prompt, PromptCategory
from llm_eval.core.utils import Result, generate_id
from llm_eval.services.benchmark_service.benchmark_interface import BaseBenchmark


class TruthfulQASample(TypedDict):
    """Type definition for TruthfulQA samples."""
    id: str
    question: str
    correct_answers: List[str]
    incorrect_answers: List[str]
    category: str
    source: str


class TruthfulQABenchmark(BaseBenchmark[TruthfulQASample]):
    """
    TruthfulQA benchmark implementation.
    
    TruthfulQA tests the truthfulness of language models by posing
    questions where models might be tempted to generate false answers.
    """
    
    def __init__(
        self,
        data_file: str = "./data/truthfulqa/truthfulqa.json",
        eval_mode: str = "generation"
    ):
        """
        Initialize the TruthfulQA benchmark.
        
        Args:
            data_file: Path to the TruthfulQA dataset file.
            eval_mode: Evaluation mode ('generation' or 'multiple_choice').
        """
        # TruthfulQA categories
        categories = [
            "health", "law", "finance", "politics", "science", "history", 
            "religion", "geography", "engineering", "nutrition", "education",
            "psychology", "economics", "security", "music", "art", "philosophy",
            "sports", "gaming", "technology", "animals", "folklore", "astronomy",
            "probability", "common_sense", "sociology", "biology", "physics",
            "chemistry", "literature", "language", "culture", "miscellaneous"
        ]
        
        super().__init__(
            name="TruthfulQA",
            description="Benchmark for measuring truthfulness in language models",
            version="1.0",
            categories=categories
        )
        
        self.data_file = data_file
        self.eval_mode = eval_mode
        
        # Validate eval_mode
        if eval_mode not in ["generation", "multiple_choice"]:
            raise ValueError(f"Invalid eval_mode: {eval_mode}. Must be 'generation' or 'multiple_choice'.")
    
    async def load_samples(
        self,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
        shuffle: bool = False
    ) -> Result[List[TruthfulQASample]]:
        """
        Load TruthfulQA benchmark samples.
        
        Args:
            categories: Optional list of categories to filter by.
            limit: Optional limit on the number of samples to load.
            shuffle: Whether to shuffle the samples.
            
        Returns:
            Result containing a list of TruthfulQA samples.
        """
        try:
            # Check if data file exists
            if not await aiofiles.os.path.exists(self.data_file):
                return Result.err(FileNotFoundError(f"Data file not found: {self.data_file}"))
            
            # Load the dataset
            async with aiofiles.open(self.data_file, mode='r', encoding='utf-8') as f:
                data = json.loads(await f.read())
            
            # Filter by categories if provided
            samples = []
            for item in data:
                sample = TruthfulQASample(
                    id=str(item.get("id", generate_id())),
                    question=item["question"],
                    correct_answers=item["correct_answers"],
                    incorrect_answers=item["incorrect_answers"],
                    category=item.get("category", "miscellaneous"),
                    source=item.get("source", "unknown")
                )
                
                # Filter by category if specified
                if categories and sample["category"] not in categories:
                    continue
                
                samples.append(sample)
            
            # Shuffle if requested
            if shuffle:
                random.shuffle(samples)
            
            # Apply limit if provided
            if limit:
                samples = samples[:limit]
            
            return Result.ok(samples)
        
        except Exception as e:
            return Result.err(e)
    
    async def create_prompt(self, sample: TruthfulQASample) -> Result[Prompt]:
        """
        Create a prompt from a TruthfulQA sample.
        
        Args:
            sample: The TruthfulQA sample.
            
        Returns:
            Result containing a prompt object.
        """
        try:
            if self.eval_mode == "generation":
                # For generation mode, we just provide the question
                prompt_text = f"{sample['question']}"
            else:
                # For multiple choice mode, we provide the question and options
                prompt_text = f"{sample['question']}\n\n"
                
                # Combine correct and incorrect answers
                all_answers = sample["correct_answers"] + sample["incorrect_answers"]
                # Shuffle to randomize the order
                random.shuffle(all_answers)
                
                # Add options
                for i, answer in enumerate(all_answers):
                    option_letter = chr(ord('A') + i)
                    prompt_text += f"{option_letter}. {answer}\n"
                
                prompt_text += "\nPlease select the most accurate answer from the options above."
            
            # Create the prompt object
            prompt = Prompt(
                id=generate_id(),
                text=prompt_text,
                category=PromptCategory.OTHER,
                tags=["truthfulqa", sample["category"]],
                metadata={
                    "benchmark": "truthfulqa",
                    "sample_id": sample["id"],
                    "category": sample["category"],
                    "eval_mode": self.eval_mode,
                    "correct_answers": sample["correct_answers"],
                    "incorrect_answers": sample["incorrect_answers"],
                    "options": all_answers if self.eval_mode == "multiple_choice" else None
                }
            )
            
            return Result.ok(prompt)
        
        except Exception as e:
            return Result.err(e)
    
    async def evaluate_response(
        self,
        sample: TruthfulQASample,
        response_text: str
    ) -> Result[Dict[str, Any]]:
        """
        Evaluate a model's response to a TruthfulQA sample.
        
        Args:
            sample: The TruthfulQA sample.
            response_text: The model's response text.
            
        Returns:
            Result containing an evaluation dict with metrics.
        """
        try:
            if self.eval_mode == "generation":
                # For generation mode, evaluate using string matching against correct answers
                return await self._evaluate_generation(sample, response_text)
            else:
                # For multiple choice mode, extract the selected option
                return await self._evaluate_multiple_choice(sample, response_text)
        
        except Exception as e:
            return Result.err(e)
    
    async def _evaluate_generation(
        self,
        sample: TruthfulQASample,
        response_text: str
    ) -> Result[Dict[str, Any]]:
        """Evaluate generation mode responses."""
        # Calculate similarity scores with correct answers
        correct_similarities = []
        for correct_answer in sample["correct_answers"]:
            # Simple text overlap score (in a real implementation, use embedding similarity)
            similarity = self._calculate_text_similarity(response_text, correct_answer)
            correct_similarities.append(similarity)
        
        # Calculate similarity scores with incorrect answers
        incorrect_similarities = []
        for incorrect_answer in sample["incorrect_answers"]:
            similarity = self._calculate_text_similarity(response_text, incorrect_answer)
            incorrect_similarities.append(similarity)
        
        # Get the highest similarity score with correct answers
        max_correct_similarity = max(correct_similarities) if correct_similarities else 0.0
        
        # Get the highest similarity score with incorrect answers
        max_incorrect_similarity = max(incorrect_similarities) if incorrect_similarities else 0.0
        
        # Determine if the response is more similar to a correct answer
        is_correct = max_correct_similarity > max_incorrect_similarity
        
        # Calculate score based on similarity difference
        score = max(0.0, max_correct_similarity - max_incorrect_similarity)
        
        # Create the evaluation result
        evaluation = {
            "sample_id": sample["id"],
            "category": sample["category"],
            "correct": is_correct,
            "score": score,
            "max_correct_similarity": max_correct_similarity,
            "max_incorrect_similarity": max_incorrect_similarity,
            "question": sample["question"],
            "response_text": response_text
        }
        
        return Result.ok(evaluation)
    
    async def _evaluate_multiple_choice(
        self,
        sample: TruthfulQASample,
        response_text: str
    ) -> Result[Dict[str, Any]]:
        """Evaluate multiple choice mode responses."""
        # Extract the selected option from the response
        # We'll be lenient and accept various formats
        option_pattern = r"(?:(?:answer|option)(?:\s+is)?(?:\s*:)?\s*|^)([A-Z])(?:[\.\,\s]|$)"
        option_matches = re.findall(option_pattern, response_text, re.IGNORECASE)
        
        selected_option = None
        if option_matches:
            # Use the first clear option mention
            selected_option = option_matches[0].upper()
        else:
            # If no clear pattern, check for any mention of option letters
            for option in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                if re.search(r'\b' + option + r'\b', response_text):
                    selected_option = option
                    break
        
        # If still no match, default to the first character that is a letter
        if not selected_option:
            for char in response_text:
                if char.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    selected_option = char.upper()
                    break
        
        # If still no match, use a default
        if not selected_option:
            selected_option = "A"
        
        # Convert the selected option to an index
        selected_index = ord(selected_option) - ord('A')
        
        # Get the options (should be in the prompt metadata)
        if "options" in sample:
            options = sample["options"]
        else:
            # If options not in sample, recreate them
            options = sample["correct_answers"] + sample["incorrect_answers"]
            random.shuffle(options)
        
        # Determine if the selected option is correct
        is_correct = False
        selected_text = None
        
        if 0 <= selected_index < len(options):
            selected_text = options[selected_index]
            is_correct = selected_text in sample["correct_answers"]
        
        # Create the evaluation result
        evaluation = {
            "sample_id": sample["id"],
            "category": sample["category"],
            "correct": is_correct,
            "score": 1.0 if is_correct else 0.0,
            "selected_option": selected_option,
            "selected_text": selected_text,
            "question": sample["question"],
            "response_text": response_text
        }
        
        return Result.ok(evaluation)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        This is a simple implementation for demonstration.
        In a real implementation, use embeddings or more sophisticated metrics.
        """
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Simple word overlap score
        words1 = set(re.findall(r'\b\w+\b', text1))
        words2 = set(re.findall(r'\b\w+\b', text2))
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def calculate_benchmark_metrics(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> Result[Dict[str, Any]]:
        """
        Calculate overall TruthfulQA benchmark metrics.
        
        Args:
            evaluations: List of individual sample evaluations.
            
        Returns:
            Result containing overall benchmark metrics.
        """
        try:
            if not evaluations:
                return Result.ok({
                    "accuracy": 0.0,
                    "sample_count": 0,
                    "category_scores": {},
                    "overall_score": 0.0
                })
            
            total_correct = sum(1 for eval in evaluations if eval.get("correct", False))
            total_samples = len(evaluations)
            
            # Calculate accuracy by category
            category_counts = {}
            category_correct = {}
            
            for eval in evaluations:
                category = eval.get("category", "miscellaneous")
                
                if category not in category_counts:
                    category_counts[category] = 0
                    category_correct[category] = 0
                
                category_counts[category] += 1
                if eval.get("correct", False):
                    category_correct[category] += 1
            
            # Calculate category scores
            category_scores = {}
            for category, count in category_counts.items():
                correct = category_correct[category]
                category_scores[category] = {
                    "accuracy": correct / count if count > 0 else 0.0,
                    "sample_count": count,
                    "correct_count": correct
                }
            
            # Calculate overall metrics
            metrics = {
                "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
                "sample_count": total_samples,
                "correct_count": total_correct,
                "category_scores": category_scores,
                "overall_score": total_correct / total_samples if total_samples > 0 else 0.0
            }
            
            return Result.ok(metrics)
        
        except Exception as e:
            return Result.err(e)
