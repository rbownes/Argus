"""
Implementation of the MMLU (Massive Multitask Language Understanding) benchmark.
"""
import json
import os
import random
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Union
import csv
import aiofiles
import aiofiles.os
import re

from llm_eval.core.models import Prompt, PromptCategory
from llm_eval.core.utils import Result, generate_id
from llm_eval.services.benchmark_service.benchmark_interface import BaseBenchmark


class MMLUSample(TypedDict):
    """Type definition for MMLU samples."""
    question: str
    options: List[str]
    answer: str
    subject: str
    id: str


class MMLUBenchmark(BaseBenchmark[MMLUSample]):
    """
    Massive Multitask Language Understanding (MMLU) benchmark implementation.
    
    MMLU covers 57 subjects spanning STEM, humanities, social sciences, and more.
    It tests both world knowledge and problem-solving ability.
    """
    
    def __init__(
        self,
        data_dir: str = "./data/mmlu",
        few_shot_examples: int = 5
    ):
        """
        Initialize the MMLU benchmark.
        
        Args:
            data_dir: Directory containing MMLU data files.
            few_shot_examples: Number of few-shot examples to include in prompts.
        """
        # Default MMLU categories (subjects)
        categories = [
            # STEM
            "abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_physics", "computer_security",
            "conceptual_physics", "electrical_engineering", "elementary_mathematics", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science", "high_school_mathematics",
            "high_school_physics", "high_school_statistics", "machine_learning",
            # Humanities
            "formal_logic", "high_school_european_history", "high_school_geography",
            "high_school_government_and_politics", "high_school_macroeconomics", "high_school_microeconomics",
            "high_school_psychology", "high_school_us_history", "high_school_world_history",
            "international_law", "jurisprudence", "logical_fallacies", "moral_disputes", "moral_scenarios",
            "philosophy", "prehistory", "professional_law", "world_religions",
            # Social Sciences
            "econometrics", "management", "marketing", "medical_genetics", "miscellaneous",
            "nutrition", "professional_accounting", "professional_medicine", "sociology", "us_foreign_policy",
            # Other
            "global_facts", "human_aging", "human_sexuality", "professional_psychology",
            "public_relations", "security_studies", "virology"
        ]
        
        super().__init__(
            name="MMLU",
            description="Massive Multitask Language Understanding Benchmark",
            version="1.0",
            categories=categories
        )
        
        self.data_dir = data_dir
        self.few_shot_examples = few_shot_examples
        
        # Map subject names to file paths
        self.subject_files = {}
        for subject in categories:
            self.subject_files[subject] = {
                "dev": os.path.join(data_dir, f"{subject}_dev.csv"),
                "test": os.path.join(data_dir, f"{subject}_test.csv"),
                "val": os.path.join(data_dir, f"{subject}_val.csv")
            }
    
    async def load_samples(
        self,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
        shuffle: bool = False,
        split: str = "test"
    ) -> Result[List[MMLUSample]]:
        """
        Load MMLU benchmark samples.
        
        Args:
            categories: Optional list of subjects to filter by.
            limit: Optional limit on the number of samples to load.
            shuffle: Whether to shuffle the samples.
            split: Which data split to use ('dev', 'test', or 'val').
            
        Returns:
            Result containing a list of MMLU samples.
        """
        try:
            # Filter categories if provided
            subjects = categories if categories else self.categories
            
            # Validate subjects
            for subject in subjects:
                if subject not in self.categories:
                    return Result.err(ValueError(f"Invalid subject: {subject}"))
            
            # Validate split
            if split not in ["dev", "test", "val"]:
                return Result.err(ValueError(f"Invalid split: {split}. Must be 'dev', 'test', or 'val'."))
            
            # Check if data directory exists
            if not await aiofiles.os.path.exists(self.data_dir):
                return Result.err(FileNotFoundError(f"Data directory not found: {self.data_dir}"))
            
            samples = []
            
            # Load samples from each subject
            for subject in subjects:
                file_path = self.subject_files[subject][split]
                
                if not await aiofiles.os.path.exists(file_path):
                    continue
                
                # Read samples from CSV file
                async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
                    # Use TextIOWrapper to get a file-like object for csv.reader
                    content = await f.read()
                    csv_reader = csv.reader(content.splitlines())
                    
                    for row in csv_reader:
                        if len(row) < 5:  # Question + 4 options
                            continue
                        
                        question = row[0]
                        options = row[1:5]  # A, B, C, D
                        
                        # The answer is the index (0-3) of the correct option
                        # We need to convert it to A, B, C, D
                        try:
                            answer = row[5].strip().upper() if len(row) > 5 else 'A'
                            if answer not in ['A', 'B', 'C', 'D']:
                                answer = 'A'  # Default if answer is invalid
                        except (IndexError, AttributeError):
                            answer = 'A'  # Default if there's any error
                        
                        sample = MMLUSample(
                            id=generate_id(),
                            question=question,
                            options=options,
                            answer=answer,
                            subject=subject
                        )
                        
                        samples.append(sample)
                        
                        # Break if we've reached the limit
                        if limit and len(samples) >= limit:
                            break
                
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
    
    async def load_few_shot_examples(
        self,
        subject: str,
        split: str = "dev",
        count: int = 5
    ) -> Result[List[MMLUSample]]:
        """
        Load few-shot examples for a subject.
        
        Args:
            subject: The subject to load examples for.
            split: Which data split to use for examples.
            count: Number of examples to load.
            
        Returns:
            Result containing few-shot examples.
        """
        # We'll use the dev split for few-shot examples
        examples_result = await self.load_samples(
            categories=[subject],
            limit=count,
            shuffle=True,
            split=split
        )
        
        if examples_result.is_err:
            return examples_result
        
        examples = examples_result.unwrap()
        
        # If we don't have enough examples, try using examples from other subjects
        if len(examples) < count:
            # Get random subjects excluding the current one
            other_subjects = [s for s in self.categories if s != subject]
            random.shuffle(other_subjects)
            
            for other_subject in other_subjects:
                if len(examples) >= count:
                    break
                
                other_examples_result = await self.load_samples(
                    categories=[other_subject],
                    limit=count - len(examples),
                    shuffle=True,
                    split=split
                )
                
                if other_examples_result.is_ok:
                    examples.extend(other_examples_result.unwrap())
        
        return Result.ok(examples[:count])
    
    async def create_prompt(
        self,
        sample: MMLUSample,
        include_few_shot: bool = True
    ) -> Result[Prompt]:
        """
        Create a prompt from an MMLU sample.
        
        Args:
            sample: The MMLU sample.
            include_few_shot: Whether to include few-shot examples.
            
        Returns:
            Result containing a prompt object.
        """
        try:
            prompt_text = ""
            
            # Add few-shot examples if requested
            if include_few_shot and self.few_shot_examples > 0:
                examples_result = await self.load_few_shot_examples(
                    subject=sample["subject"],
                    count=self.few_shot_examples
                )
                
                if examples_result.is_ok:
                    examples = examples_result.unwrap()
                    
                    for i, example in enumerate(examples):
                        prompt_text += f"Question: {example['question']}\n"
                        for j, option in enumerate(example['options']):
                            option_letter = chr(ord('A') + j)
                            prompt_text += f"{option_letter}. {option}\n"
                        prompt_text += f"Answer: {example['answer']}\n\n"
            
            # Add the actual question
            prompt_text += f"Question: {sample['question']}\n"
            for i, option in enumerate(sample['options']):
                option_letter = chr(ord('A') + i)
                prompt_text += f"{option_letter}. {option}\n"
            prompt_text += "Answer:"
            
            # Create the prompt object
            prompt = Prompt(
                id=generate_id(),
                text=prompt_text,
                category=PromptCategory.OTHER,
                tags=["mmlu", sample["subject"]],
                metadata={
                    "benchmark": "mmlu",
                    "subject": sample["subject"],
                    "sample_id": sample["id"],
                    "expected_answer": sample["answer"]
                }
            )
            
            return Result.ok(prompt)
        
        except Exception as e:
            return Result.err(e)
    
    async def evaluate_response(
        self,
        sample: MMLUSample,
        response_text: str
    ) -> Result[Dict[str, Any]]:
        """
        Evaluate a model's response to an MMLU sample.
        
        Args:
            sample: The MMLU sample.
            response_text: The model's response text.
            
        Returns:
            Result containing an evaluation dict with metrics.
        """
        try:
            # Extract the answer from the response
            # We'll be lenient and accept various formats
            
            # First, try to find a clear answer pattern like "The answer is A" or just "A."
            answer_pattern = r"(?:(?:answer|option)(?:\s+is)?(?:\s*:)?\s*|^)([A-D])(?:[\.\,\s]|$)"
            answer_matches = re.findall(answer_pattern, response_text, re.IGNORECASE)
            
            predicted_answer = None
            if answer_matches:
                # Use the first clear answer mention
                predicted_answer = answer_matches[0].upper()
            else:
                # If no clear pattern, check for any mention of A, B, C, or D
                for option in ["A", "B", "C", "D"]:
                    if re.search(r'\b' + option + r'\b', response_text):
                        predicted_answer = option
                        break
            
            # If still no match, default to the first character that is A, B, C, or D
            if not predicted_answer:
                for char in response_text:
                    if char.upper() in ["A", "B", "C", "D"]:
                        predicted_answer = char.upper()
                        break
            
            # If still no match, use a default
            if not predicted_answer:
                predicted_answer = "A"
            
            # Check if the prediction is correct
            correct = predicted_answer == sample["answer"]
            
            # Create the evaluation result
            evaluation = {
                "sample_id": sample["id"],
                "subject": sample["subject"],
                "correct": correct,
                "score": 1.0 if correct else 0.0,
                "expected_answer": sample["answer"],
                "predicted_answer": predicted_answer,
                "question": sample["question"],
                "options": sample["options"],
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
        Calculate overall MMLU benchmark metrics.
        
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
                    "subject_scores": {},
                    "overall_score": 0.0
                })
            
            total_correct = sum(1 for eval in evaluations if eval["correct"])
            total_samples = len(evaluations)
            
            # Calculate accuracy by subject
            subject_counts = {}
            subject_correct = {}
            
            for eval in evaluations:
                subject = eval["subject"]
                
                if subject not in subject_counts:
                    subject_counts[subject] = 0
                    subject_correct[subject] = 0
                
                subject_counts[subject] += 1
                if eval["correct"]:
                    subject_correct[subject] += 1
            
            # Calculate subject scores
            subject_scores = {}
            for subject, count in subject_counts.items():
                correct = subject_correct[subject]
                subject_scores[subject] = {
                    "accuracy": correct / count if count > 0 else 0.0,
                    "sample_count": count,
                    "correct_count": correct
                }
            
            # Calculate overall metrics
            metrics = {
                "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
                "sample_count": total_samples,
                "correct_count": total_correct,
                "subject_scores": subject_scores,
                "overall_score": total_correct / total_samples if total_samples > 0 else 0.0
            }
            
            return Result.ok(metrics)
        
        except Exception as e:
            return Result.err(e)
