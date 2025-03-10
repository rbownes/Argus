"""
Implementation of the GSM8K (Grade School Math 8K) benchmark.
"""
import json
import os
import re
from typing import Dict, List, Any, Optional, TypedDict, Union
import aiofiles
import random

from llm_eval.core.models import Prompt, PromptCategory
from llm_eval.core.utils import Result, generate_id
from llm_eval.services.benchmark_service.benchmark_interface import BaseBenchmark


class GSM8KSample(TypedDict):
    """Type definition for GSM8K samples."""
    id: str
    question: str
    answer: str
    solution: str
    difficulty: Optional[str]


class GSM8KBenchmark(BaseBenchmark[GSM8KSample]):
    """
    GSM8K (Grade School Math 8K) benchmark implementation.
    
    GSM8K is a dataset of grade school math word problems requiring multi-step
    reasoning to solve. It tests language models' ability to solve complex
    arithmetic reasoning problems.
    """
    
    def __init__(
        self,
        data_file: str = "./data/gsm8k/gsm8k.jsonl",
        few_shot_examples: int = 3
    ):
        """
        Initialize the GSM8K benchmark.
        
        Args:
            data_file: Path to the GSM8K dataset file.
            few_shot_examples: Number of few-shot examples to include in prompts.
        """
        # GSM8K doesn't have explicit categories, so we'll use difficulty levels
        categories = ["easy", "medium", "hard", "unknown"]
        
        super().__init__(
            name="GSM8K",
            description="Grade School Math 8K Benchmark",
            version="1.0",
            categories=categories
        )
        
        self.data_file = data_file
        self.few_shot_examples = few_shot_examples
    
    async def load_samples(
        self,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
        shuffle: bool = False
    ) -> Result[List[GSM8KSample]]:
        """
        Load GSM8K benchmark samples.
        
        Args:
            categories: Optional list of difficulty levels to filter by.
            limit: Optional limit on the number of samples to load.
            shuffle: Whether to shuffle the samples.
            
        Returns:
            Result containing a list of GSM8K samples.
        """
        try:
            # Check if data file exists
            if not await aiofiles.os.path.exists(self.data_file):
                return Result.err(FileNotFoundError(f"Data file not found: {self.data_file}"))
            
            # Load the dataset (JSONL format)
            samples = []
            async with aiofiles.open(self.data_file, mode='r', encoding='utf-8') as f:
                lines = await f.readlines()
                
                for line in lines:
                    if not line.strip():
                        continue
                    
                    item = json.loads(line)
                    
                    # Extract answer from solution
                    answer = None
                    solution = item.get("answer", "")
                    
                    # Try to extract the final numeric answer
                    answer_match = re.search(r'The answer is (\d+)', solution)
                    if answer_match:
                        answer = answer_match.group(1)
                    else:
                        # Try to extract just the last number
                        numbers = re.findall(r'(\d+)', solution)
                        if numbers:
                            answer = numbers[-1]
                        else:
                            answer = "Unknown"
                    
                    # Determine difficulty (not in the original dataset, but we can estimate)
                    difficulty = "unknown"
                    if solution:
                        step_count = len(re.findall(r'[\.\n]', solution))
                        if step_count <= 2:
                            difficulty = "easy"
                        elif step_count <= 5:
                            difficulty = "medium"
                        else:
                            difficulty = "hard"
                    
                    sample = GSM8KSample(
                        id=str(item.get("id", generate_id())),
                        question=item["question"],
                        answer=answer,
                        solution=solution,
                        difficulty=difficulty
                    )
                    
                    # Filter by category (difficulty) if specified
                    if categories and sample["difficulty"] not in categories:
                        continue
                    
                    samples.append(sample)
                    
                    # Break if we've reached the limit
                    if limit and len(samples) >= limit:
                        break
            
            # Shuffle if requested
            if shuffle:
                random.shuffle(samples)
            
            # Apply limit if provided
            if limit:
                samples = samples[:limit]
            
            return Result.ok(samples)
        
        except Exception as e:
            return Result.err(e)
    
    async def load_few_shot_examples(self, count: int = 3) -> Result[List[GSM8KSample]]:
        """
        Load few-shot examples.
        
        Args:
            count: Number of examples to load.
            
        Returns:
            Result containing few-shot examples.
        """
        # Use samples from the dataset as few-shot examples
        examples_result = await self.load_samples(
            limit=count,
            shuffle=True
        )
        
        if examples_result.is_err:
            return examples_result
        
        examples = examples_result.unwrap()
        
        return Result.ok(examples[:count])
    
    async def create_prompt(
        self,
        sample: GSM8KSample,
        include_few_shot: bool = True
    ) -> Result[Prompt]:
        """
        Create a prompt from a GSM8K sample.
        
        Args:
            sample: The GSM8K sample.
            include_few_shot: Whether to include few-shot examples.
            
        Returns:
            Result containing a prompt object.
        """
        try:
            prompt_text = "Solve the following math problem step by step:\n\n"
            
            # Add few-shot examples if requested
            if include_few_shot and self.few_shot_examples > 0:
                examples_result = await self.load_few_shot_examples(
                    count=self.few_shot_examples
                )
                
                if examples_result.is_ok:
                    examples = examples_result.unwrap()
                    
                    for i, example in enumerate(examples):
                        prompt_text += f"Problem: {example['question']}\n"
                        prompt_text += f"Solution: {example['solution']}\n\n"
            
            # Add the actual question
            prompt_text += f"Problem: {sample['question']}\n"
            prompt_text += "Solution:"
            
            # Create the prompt object
            prompt = Prompt(
                id=generate_id(),
                text=prompt_text,
                category=PromptCategory.OTHER,
                tags=["gsm8k", f"difficulty_{sample['difficulty']}"],
                metadata={
                    "benchmark": "gsm8k",
                    "sample_id": sample["id"],
                    "difficulty": sample["difficulty"],
                    "expected_answer": sample["answer"]
                }
            )
            
            return Result.ok(prompt)
        
        except Exception as e:
            return Result.err(e)
    
    async def evaluate_response(
        self,
        sample: GSM8KSample,
        response_text: str
    ) -> Result[Dict[str, Any]]:
        """
        Evaluate a model's response to a GSM8K sample.
        
        Args:
            sample: The GSM8K sample.
            response_text: The model's response text.
            
        Returns:
            Result containing an evaluation dict with metrics.
        """
        try:
            # Look for explicit answer formatting like "The answer is X" or "Therefore, X is the answer"
            answer_patterns = [
                r"The answer is[:\s]*(\d+)",
                r"Therefore,? (?:the answer is )?(\d+)",
                r"Thus,? (?:the answer is )?(\d+)",
                r"So,? (?:the answer is )?(\d+)",
                r"Hence,? (?:the answer is )?(\d+)",
                r"= (\d+)$",
                r"= (\d+)[\.!\s]"
            ]
            
            predicted_answer = None
            for pattern in answer_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    predicted_answer = match.group(1)
                    break
            
            # If no explicit answer format found, look for the last number in the response
            if not predicted_answer:
                numbers = re.findall(r'(\d+)', response_text)
                if numbers:
                    predicted_answer = numbers[-1]
                else:
                    predicted_answer = "0"  # Default if no numbers found
            
            # Remove commas and whitespace from answers for comparison
            expected_answer_clean = re.sub(r'[,\s]', '', sample["answer"])
            predicted_answer_clean = re.sub(r'[,\s]', '', predicted_answer)
            
            # Check if the prediction is correct
            correct = predicted_answer_clean == expected_answer_clean
            
            # Look for reasoning steps in the response
            steps = re.split(r'[\.\n]', response_text)
            steps = [s.strip() for s in steps if s.strip()]
            
            # Create the evaluation result
            evaluation = {
                "sample_id": sample["id"],
                "difficulty": sample["difficulty"],
                "correct": correct,
                "score": 1.0 if correct else 0.0,
                "expected_answer": sample["answer"],
                "predicted_answer": predicted_answer,
                "question": sample["question"],
                "step_count": len(steps),
                "response_text": response_text
            }
            
            return Result.ok(evaluation)
        
        except Exception as e:
            return Result.err(e)
    
    async def calculate_benchmark_metrics(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> Result[Dict[str, Any]]:
        """
        Calculate overall GSM8K benchmark metrics.
        
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
                    "difficulty_scores": {},
                    "overall_score": 0.0
                })
            
            total_correct = sum(1 for eval in evaluations if eval["correct"])
            total_samples = len(evaluations)
            
            # Calculate accuracy by difficulty
            difficulty_counts = {}
            difficulty_correct = {}
            
            for eval in evaluations:
                difficulty = eval["difficulty"]
                
                if difficulty not in difficulty_counts:
                    difficulty_counts[difficulty] = 0
                    difficulty_correct[difficulty] = 0
                
                difficulty_counts[difficulty] += 1
                if eval["correct"]:
                    difficulty_correct[difficulty] += 1
            
            # Calculate difficulty scores
            difficulty_scores = {}
            for difficulty, count in difficulty_counts.items():
                correct = difficulty_correct[difficulty]
                difficulty_scores[difficulty] = {
                    "accuracy": correct / count if count > 0 else 0.0,
                    "sample_count": count,
                    "correct_count": correct
                }
            
            # Calculate average step count
            avg_step_count = sum(eval.get("step_count", 0) for eval in evaluations) / total_samples if total_samples > 0 else 0
            
            # Calculate overall metrics
            metrics = {
                "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
                "sample_count": total_samples,
                "correct_count": total_correct,
                "avg_step_count": avg_step_count,
                "difficulty_scores": difficulty_scores,
                "overall_score": total_correct / total_samples if total_samples > 0 else 0.0
            }
            
            return Result.ok(metrics)
        
        except Exception as e:
            return Result.err(e)
