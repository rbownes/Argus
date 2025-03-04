"""
LLM Metrics Collection Framework

This module provides a flexible framework for collecting and computing metrics on LLM outputs
stored in ChromaDB. It supports built-in metrics, DeepEval integration, and custom metrics.
"""

from typing import Dict, List, Any, Optional, Union, Callable, Type, Protocol, Set, Tuple
import logging
import inspect
import time
import json
import datetime
import uuid
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator, ConfigDict

import chromadb
from chromadb.api.models.Collection import Collection
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import DeepEval for LLM-specific metrics
try:
    import deepeval
    from deepeval.metrics import HallucinationMetric, FactualConsistencyMetric, RelevanceMetric, CoherenceMetric
    DEEPEVAL_AVAILABLE = True
    logger.info("DeepEval is available for LLM metrics")
except ImportError:
    DEEPEVAL_AVAILABLE = False
    logger.warning("DeepEval not found. Install with 'pip install deepeval' for LLM-specific metrics")


# --- Metric Definition ---

class MetricCategory(str, Enum):
    """Categories for organizing metrics."""
    BASIC = "basic"
    LLM_QUALITY = "llm_quality"
    PERFORMANCE = "performance"
    CUSTOM = "custom"
    OPERATIONAL = "operational"


class MetricResult(BaseModel):
    """Standard structure for metric results."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    metric_name: str = Field(..., description="Name of the metric")
    batch_id: str = Field(..., description="ID of the batch being evaluated")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now, description="When the metric was computed")
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this result")
    
    # Primary result values
    value: float = Field(..., description="Primary numeric result value")
    
    # Additional information
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the metric run")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed results or breakdown")
    success: bool = Field(default=True, description="Whether the metric calculation succeeded")
    error: Optional[str] = Field(default=None, description="Error message if calculation failed")
    
    # Reference information
    model_name: Optional[str] = Field(default=None, description="Model that generated the evaluated responses")
    sample_count: int = Field(..., description="Number of samples used in the calculation")


class MetricDependency(BaseModel):
    """Defines a dependency on another metric."""
    metric_name: str = Field(..., description="Name of the required metric")
    required: bool = Field(default=True, description="Whether this dependency is required")


class MetricParams(BaseModel):
    """Parameters for metric calculation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    batch_id: str = Field(..., description="ID of the batch to calculate metrics for")
    collection: Collection = Field(..., description="ChromaDB collection containing the responses")
    model_name: Optional[str] = Field(default=None, description="Filter responses to specific model")
    max_samples: Optional[int] = Field(default=None, description="Maximum number of samples to evaluate")
    dependencies_results: Dict[str, MetricResult] = Field(default_factory=dict, 
                                               description="Results from dependency metrics")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters for the metric")


class Metric(ABC):
    """Base class for all metrics."""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.category = MetricCategory.BASIC
        self.description = inspect.getdoc(self.__class__) or "No description provided"
        self.dependencies: List[MetricDependency] = []
    
    @abstractmethod
    def compute(self, params: MetricParams) -> MetricResult:
        """
        Compute the metric on the given batch.
        
        Args:
            params: Parameters for the metric calculation
            
        Returns:
            Metric result object
        """
        pass
    
    def get_dependencies(self) -> List[MetricDependency]:
        """Get the list of dependencies for this metric."""
        return self.dependencies
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this metric."""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "dependencies": [dep.dict() for dep in self.dependencies]
        }


# --- Metric Registry ---

class MetricRegistry:
    """Registry for available metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, Type[Metric]] = {}
        self._instances: Dict[str, Metric] = {}
    
    def register(self, metric_class: Type[Metric]) -> None:
        """
        Register a metric class.
        
        Args:
            metric_class: The metric class to register
        """
        instance = metric_class()
        name = instance.name
        if name in self._metrics:
            logger.warning(f"Metric '{name}' already registered. Overwriting.")
        
        self._metrics[name] = metric_class
        self._instances[name] = instance
        logger.debug(f"Registered metric: {name}")
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """
        Get a metric instance by name.
        
        Args:
            name: Name of the metric
            
        Returns:
            Metric instance or None if not found
        """
        return self._instances.get(name)
    
    def list_metrics(self, category: Optional[MetricCategory] = None) -> List[Dict[str, Any]]:
        """
        List all registered metrics.
        
        Args:
            category: Optional filter by category
            
        Returns:
            List of metric information dictionaries
        """
        metrics = []
        for name, instance in self._instances.items():
            if category is None or instance.category == category:
                metrics.append(instance.get_info())
        return metrics
    
    def get_all_instances(self) -> Dict[str, Metric]:
        """Get all registered metric instances."""
        return self._instances.copy()


# --- Metric Pipeline ---

class MetricPipeline:
    """Pipeline for running multiple metrics on batches of data."""
    
    def __init__(self, registry: MetricRegistry):
        self.registry = registry
        self.results_history: List[Dict[str, MetricResult]] = []
    
    def run_metric(self, metric_name: str, batch_id: str, collection: Collection,
                  model_name: Optional[str] = None, max_samples: Optional[int] = None,
                  parameters: Optional[Dict[str, Any]] = None) -> MetricResult:
        """
        Run a single metric on a batch.
        
        Args:
            metric_name: Name of the metric to run
            batch_id: ID of the batch to run the metric on
            collection: ChromaDB collection containing the batch
            model_name: Optional model name to filter responses
            max_samples: Maximum number of samples to evaluate
            parameters: Additional parameters for the metric
            
        Returns:
            Metric result
            
        Raises:
            ValueError: If metric is not found or dependencies cannot be satisfied
        """
        metric = self.registry.get_metric(metric_name)
        if metric is None:
            raise ValueError(f"Metric '{metric_name}' not found in registry")
        
        # Get dependencies
        dependency_results = {}
        for dep in metric.get_dependencies():
            # Check if we've already computed this dependency
            dep_result = self._find_dependency_result(dep.metric_name, batch_id, model_name)
            
            if dep_result is None:
                # Need to compute the dependency
                try:
                    dep_result = self.run_metric(
                        dep.metric_name, batch_id, collection, 
                        model_name, max_samples, parameters
                    )
                except Exception as e:
                    if dep.required:
                        raise ValueError(f"Failed to compute required dependency '{dep.metric_name}': {e}")
                    else:
                        logger.warning(f"Failed to compute optional dependency '{dep.metric_name}': {e}")
                        continue
            
            dependency_results[dep.metric_name] = dep_result
        
        # Set up parameters
        params = MetricParams(
            batch_id=batch_id,
            collection=collection,
            model_name=model_name,
            max_samples=max_samples,
            dependencies_results=dependency_results,
            parameters=parameters or {}
        )
        
        # Compute the metric
        try:
            start_time = time.time()
            result = metric.compute(params)
            elapsed_time = time.time() - start_time
            
            # Add some metadata
            if result.metadata is None:
                result.metadata = {}
            result.metadata["computation_time_seconds"] = elapsed_time
            
            # Store the result
            self._store_result(result)
            
            return result
        
        except Exception as e:
            logger.error(f"Error computing metric '{metric_name}': {str(e)}", exc_info=True)
            
            # Create error result
            error_result = MetricResult(
                metric_name=metric_name,
                batch_id=batch_id,
                value=float('nan'),
                sample_count=0,
                model_name=model_name,
                success=False,
                error=str(e),
                metadata={"computation_time_seconds": time.time() - start_time}
            )
            self._store_result(error_result)
            
            raise
    
    def run_metrics(self, metric_names: List[str], batch_id: str, collection: Collection,
                  model_name: Optional[str] = None, max_samples: Optional[int] = None,
                  parameters: Optional[Dict[str, Any]] = None) -> Dict[str, MetricResult]:
        """
        Run multiple metrics on a batch.
        
        Args:
            metric_names: Names of metrics to run
            batch_id: ID of the batch to run the metrics on
            collection: ChromaDB collection containing the batch
            model_name: Optional model name to filter responses
            max_samples: Maximum number of samples to evaluate
            parameters: Additional parameters for metrics
            
        Returns:
            Dictionary mapping metric names to results
        """
        results = {}
        for name in metric_names:
            try:
                result = self.run_metric(name, batch_id, collection, model_name, max_samples, parameters)
                results[name] = result
            except Exception as e:
                logger.error(f"Failed to run metric '{name}': {e}")
                # Continue with other metrics even if some fail
        
        return results
    
    def run_all_metrics(self, batch_id: str, collection: Collection,
                      model_name: Optional[str] = None, max_samples: Optional[int] = None,
                      parameters: Optional[Dict[str, Any]] = None) -> Dict[str, MetricResult]:
        """
        Run all registered metrics on a batch.
        
        Args:
            batch_id: ID of the batch to run the metrics on
            collection: ChromaDB collection containing the batch
            model_name: Optional model name to filter responses
            max_samples: Maximum number of samples to evaluate
            parameters: Additional parameters for metrics
            
        Returns:
            Dictionary mapping metric names to results
        """
        all_metrics = list(self.registry.get_all_instances().keys())
        return self.run_metrics(all_metrics, batch_id, collection, model_name, max_samples, parameters)
    
    def _find_dependency_result(self, metric_name: str, batch_id: str, 
                              model_name: Optional[str] = None) -> Optional[MetricResult]:
        """Find a previously computed result for a dependency."""
        for results_batch in self.results_history:
            result = results_batch.get(metric_name)
            if (result and result.batch_id == batch_id and
                (model_name is None or result.model_name == model_name)):
                return result
        return None
    
    def _store_result(self, result: MetricResult) -> None:
        """Store a metric result."""
        # For now, we simply append to the history
        # In a real system, you would persist this to a database
        batch_results = {}
        if self.results_history:
            batch_results = self.results_history[-1]
        else:
            self.results_history.append(batch_results)
        
        batch_results[result.metric_name] = result
    
    def export_results_to_dataframe(self) -> pd.DataFrame:
        """
        Export all results to a DataFrame for analysis.
        
        Returns:
            Pandas DataFrame with all metric results
        """
        rows = []
        for batch in self.results_history:
            for name, result in batch.items():
                row = {
                    "metric_name": result.metric_name,
                    "batch_id": result.batch_id,
                    "model_name": result.model_name,
                    "timestamp": result.timestamp,
                    "value": result.value,
                    "sample_count": result.sample_count,
                    "success": result.success
                }
                # Add any metadata as columns
                for key, value in result.metadata.items():
                    if isinstance(value, (int, float, str, bool)) or value is None:
                        row[f"metadata_{key}"] = value
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_results_to_json(self, filename: str) -> None:
        """
        Save results to a JSON file.
        
        Args:
            filename: Path to save the JSON file
        """
        # Convert results to serializable format
        serializable_history = []
        for batch in self.results_history:
            serializable_batch = {}
            for name, result in batch.items():
                serializable_batch[name] = result.model_dump()
            serializable_history.append(serializable_batch)
        
        with open(filename, 'w') as f:
            json.dump(serializable_history, f, indent=2, default=str)


# --- Batch Helper Functions ---

def get_batch_responses(collection: Collection, batch_id: str, 
                       model_name: Optional[str] = None, 
                       max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get responses for a specific batch from ChromaDB.
    
    Args:
        collection: ChromaDB collection to query
        batch_id: ID of the batch to retrieve
        model_name: Optional model name to filter by
        max_samples: Maximum number of samples to retrieve
        
    Returns:
        List of response dictionaries with text and metadata
    """
    # Build the filter
    filter_dict = {"batch_id": batch_id}
    if model_name:
        filter_dict = {
            "$and": [
                {"batch_id": batch_id},
                {"model_name": model_name}
            ]
        }
    
    # Query the collection
    results = collection.get(
        where=filter_dict,
        limit=max_samples
    )
    
    # Format the results
    formatted_results = []
    if results and results['documents']:
        for i, doc in enumerate(results['documents']):
            result_entry = {
                "text": doc,
                "metadata": results['metadatas'][i] if results['metadatas'] else {},
                "id": results['ids'][i] if results['ids'] else None,
            }
            formatted_results.append(result_entry)
    
    return formatted_results


def get_batch_ids(collection: Collection) -> List[str]:
    """
    Get all batch IDs in the collection.
    
    Args:
        collection: ChromaDB collection to query
        
    Returns:
        List of unique batch IDs
    """
    results = collection.get()
    
    if not results or not results['metadatas']:
        return []
    
    # Extract unique batch IDs
    batch_ids = set()
    for metadata in results['metadatas']:
        batch_id = metadata.get('batch_id')
        if batch_id:
            batch_ids.add(batch_id)
    
    return list(batch_ids)


def get_prompts_for_batch(collection: Collection, batch_id: str) -> Dict[str, str]:
    """
    Get all unique prompts in a batch with their indices.
    
    Args:
        collection: ChromaDB collection to query
        batch_id: ID of the batch to retrieve prompts for
        
    Returns:
        Dictionary mapping prompt indices to prompt text
    """
    results = collection.get(
        where={"batch_id": batch_id}
    )
    
    prompts = {}
    if results and results['metadatas']:
        for metadata in results['metadatas']:
            prompt_idx = metadata.get('prompt_idx')
            prompt = metadata.get('prompt')
            if prompt_idx is not None and prompt:
                prompts[prompt_idx] = prompt
    
    return prompts


# --- Built-in Metrics ---

class ResponseLengthMetric(Metric):
    """Measures the average length of responses in characters."""
    
    def __init__(self):
        super().__init__()
        self.category = MetricCategory.BASIC
    
    def compute(self, params: MetricParams) -> MetricResult:
        responses = get_batch_responses(
            params.collection, params.batch_id, 
            params.model_name, params.max_samples
        )
        
        if not responses:
            return MetricResult(
                metric_name=self.name,
                batch_id=params.batch_id,
                value=0.0,
                sample_count=0,
                model_name=params.model_name,
                success=False,
                error="No responses found for batch"
            )
        
        # Calculate average length
        lengths = [len(r["text"]) for r in responses]
        avg_length = sum(lengths) / len(lengths)
        
        # Calculate additional statistics
        min_length = min(lengths)
        max_length = max(lengths)
        median_length = sorted(lengths)[len(lengths) // 2]
        
        return MetricResult(
            metric_name=self.name,
            batch_id=params.batch_id,
            value=avg_length,
            sample_count=len(responses),
            model_name=params.model_name,
            details={
                "min_length": min_length,
                "max_length": max_length,
                "median_length": median_length,
                "length_distribution": lengths
            }
        )


class TokenRatioMetric(Metric):
    """Estimates the ratio of tokens to characters in responses."""
    
    def __init__(self):
        super().__init__()
        self.category = MetricCategory.BASIC
        self.dependencies = [
            MetricDependency(metric_name="ResponseLengthMetric", required=True)
        ]
    
    def compute(self, params: MetricParams) -> MetricResult:
        responses = get_batch_responses(
            params.collection, params.batch_id, 
            params.model_name, params.max_samples
        )
        
        if not responses:
            return MetricResult(
                metric_name=self.name,
                batch_id=params.batch_id,
                value=0.0,
                sample_count=0,
                model_name=params.model_name,
                success=False,
                error="No responses found for batch"
            )
        
        # Simple token estimation (words separated by spaces)
        token_counts = []
        char_counts = []
        
        for r in responses:
            text = r["text"]
            # Simple word count as token approximation
            tokens = len(text.split())
            chars = len(text)
            
            token_counts.append(tokens)
            char_counts.append(chars)
        
        # Calculate ratio
        if sum(char_counts) == 0:
            token_char_ratio = 0
        else:
            token_char_ratio = sum(token_counts) / sum(char_counts)
        
        return MetricResult(
            metric_name=self.name,
            batch_id=params.batch_id,
            value=token_char_ratio,
            sample_count=len(responses),
            model_name=params.model_name,
            details={
                "total_tokens": sum(token_counts),
                "total_chars": sum(char_counts),
                "avg_tokens_per_response": sum(token_counts) / len(token_counts)
            }
        )


# --- DeepEval Integration (if available) ---

if DEEPEVAL_AVAILABLE:
    class HallucinationMetricAdapter(Metric):
        """Measures hallucination in responses using DeepEval."""
        
        def __init__(self):
            super().__init__()
            self.category = MetricCategory.LLM_QUALITY
        
        def compute(self, params: MetricParams) -> MetricResult:
            responses = get_batch_responses(
                params.collection, params.batch_id, 
                params.model_name, params.max_samples
            )
            
            if not responses:
                return MetricResult(
                    metric_name=self.name,
                    batch_id=params.batch_id,
                    value=0.0,
                    sample_count=0,
                    model_name=params.model_name,
                    success=False,
                    error="No responses found for batch"
                )
            
            # Get prompts for context
            prompts_dict = get_prompts_for_batch(params.collection, params.batch_id)
            
            # Set up DeepEval metric
            hallucination_metric = HallucinationMetric()
            
            # Track per-response scores
            scores = []
            details = []
            
            # Evaluate each response
            for response in responses:
                prompt_idx = response["metadata"].get("prompt_idx")
                prompt = prompts_dict.get(prompt_idx, "")
                
                # DeepEval requires a context, so we use the prompt as minimal context
                # In a real system, you would use the actual context or retrieved documents
                hallucination_metric.measure(
                    context=prompt,
                    response=response["text"]
                )
                
                # Get scores (higher is better in DeepEval, 0-1 range)
                # We invert to make lower better (standard for hallucination)
                score = 1.0 - hallucination_metric.score
                scores.append(score)
                
                details.append({
                    "id": response["id"],
                    "hallucination_score": score,
                    "raw_deepeval_score": hallucination_metric.score
                })
            
            # Calculate average score
            avg_score = sum(scores) / len(scores) if scores else 0
            
            return MetricResult(
                metric_name=self.name,
                batch_id=params.batch_id,
                value=avg_score,  # Lower is better for hallucination
                sample_count=len(responses),
                model_name=params.model_name,
                details={
                    "per_response_details": details,
                    "score_distribution": scores
                }
            )


# --- Default Registry Setup ---

def create_default_registry() -> MetricRegistry:
    """Create and populate a default metric registry."""
    registry = MetricRegistry()
    
    # Register built-in metrics
    registry.register(ResponseLengthMetric)
    registry.register(TokenRatioMetric)
    
    # Register DeepEval metrics if available
    if DEEPEVAL_AVAILABLE:
        registry.register(HallucinationMetricAdapter)
    
    return registry


# --- Example Custom Metric ---

class PositivityScoreMetric(Metric):
    """
    Custom metric that measures the positivity of responses using a simple
    positive/negative word counting heuristic.
    """
    
    def __init__(self):
        super().__init__()
        self.category = MetricCategory.CUSTOM
        
        # Simple positive and negative word lists
        self.positive_words = {
            "good", "great", "excellent", "best", "better", "positive",
            "wonderful", "fantastic", "amazing", "awesome", "outstanding",
            "superb", "brilliant", "exceptional", "favorable", "superior",
            "lovely", "terrific", "beneficial", "perfect", "remarkable"
        }
        
        self.negative_words = {
            "bad", "worst", "terrible", "awful", "horrible", "negative",
            "poor", "disappointing", "unsatisfactory", "inferior", "dreadful",
            "unfavorable", "atrocious", "appalling", "inadequate", "unpleasant",
            "abysmal", "subpar", "deficient", "mediocre", "problematic"
        }
    
    def compute(self, params: MetricParams) -> MetricResult:
        responses = get_batch_responses(
            params.collection, params.batch_id, 
            params.model_name, params.max_samples
        )
        
        if not responses:
            return MetricResult(
                metric_name=self.name,
                batch_id=params.batch_id,
                value=0.0,
                sample_count=0,
                model_name=params.model_name,
                success=False,
                error="No responses found for batch"
            )
        
        # Process each response
        scores = []
        details = []
        
        for response in responses:
            text = response["text"].lower()
            words = set(text.split())
            
            # Count positive and negative words
            positive_count = sum(1 for word in words if word in self.positive_words)
            negative_count = sum(1 for word in words if word in self.negative_words)
            
            # Calculate positivity score (-1 to +1 range)
            total = positive_count + negative_count
            score = 0
            if total > 0:
                score = (positive_count - negative_count) / total
            
            scores.append(score)
            details.append({
                "id": response["id"],
                "positivity_score": score,
                "positive_words": positive_count,
                "negative_words": negative_count
            })
        
        # Calculate average score
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return MetricResult(
            metric_name=self.name,
            batch_id=params.batch_id,
            value=avg_score,
            sample_count=len(responses),
            model_name=params.model_name,
            details={
                "per_response_details": details,
                "score_distribution": scores
            }
        )


# --- Putting it all together ---

def create_metric_pipeline() -> Tuple[MetricRegistry, MetricPipeline]:
    """Create and set up a complete metric pipeline with default metrics."""
    registry = create_default_registry()
    
    # Register the custom metric
    registry.register(PositivityScoreMetric)
    
    # Create pipeline
    pipeline = MetricPipeline(registry)
    
    return registry, pipeline