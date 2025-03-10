"""
Interface for benchmark implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TypeVar, Generic, Tuple

from llm_eval.core.models import Prompt
from llm_eval.core.utils import Result


BenchmarkResult = Dict[str, Any]
T = TypeVar('T')


class BenchmarkInterface(ABC, Generic[T]):
    """
    Base interface for implementing benchmarks.
    
    Generic type T represents the sample type for the specific benchmark.
    For example, for MMLU, T might be a dict with question, options, and answer.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the benchmark."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Get a description of the benchmark."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Get the version of the benchmark."""
        pass
    
    @property
    @abstractmethod
    def categories(self) -> List[str]:
        """Get the categories or subject areas covered by the benchmark."""
        pass
    
    @abstractmethod
    async def load_samples(
        self,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
        shuffle: bool = False
    ) -> Result[List[T]]:
        """
        Load benchmark samples.
        
        Args:
            categories: Optional list of categories to filter by.
            limit: Optional limit on the number of samples to load.
            shuffle: Whether to shuffle the samples.
            
        Returns:
            Result containing a list of benchmark samples.
        """
        pass
    
    @abstractmethod
    async def create_prompt(self, sample: T) -> Result[Prompt]:
        """
        Create a prompt from a benchmark sample.
        
        Args:
            sample: The benchmark sample.
            
        Returns:
            Result containing a prompt object.
        """
        pass
    
    @abstractmethod
    async def evaluate_response(
        self,
        sample: T,
        response_text: str
    ) -> Result[Dict[str, Any]]:
        """
        Evaluate a model's response to a benchmark sample.
        
        Args:
            sample: The benchmark sample.
            response_text: The model's response text.
            
        Returns:
            Result containing an evaluation dict with metrics.
        """
        pass
    
    @abstractmethod
    async def calculate_benchmark_metrics(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> Result[Dict[str, Any]]:
        """
        Calculate overall benchmark metrics from individual evaluations.
        
        Args:
            evaluations: List of individual sample evaluations.
            
        Returns:
            Result containing overall benchmark metrics.
        """
        pass


class BaseBenchmark(BenchmarkInterface[T]):
    """Base implementation with common functionality for benchmarks."""
    
    def __init__(
        self,
        name: str,
        description: str,
        version: str,
        categories: List[str]
    ):
        """
        Initialize the base benchmark.
        
        Args:
            name: Name of the benchmark.
            description: Description of the benchmark.
            version: Version of the benchmark.
            categories: Categories or subject areas in the benchmark.
        """
        self._name = name
        self._description = description
        self._version = version
        self._categories = categories
    
    @property
    def name(self) -> str:
        """Get the name of the benchmark."""
        return self._name
    
    @property
    def description(self) -> str:
        """Get a description of the benchmark."""
        return self._description
    
    @property
    def version(self) -> str:
        """Get the version of the benchmark."""
        return self._version
    
    @property
    def categories(self) -> List[str]:
        """Get the categories or subject areas covered by the benchmark."""
        return self._categories
