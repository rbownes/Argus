"""
Utility functions for the LLM Evaluation Framework.
"""
import uuid
import time
from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable, Union

T = TypeVar('T')


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


def measure_latency(func: Callable[..., T]) -> Callable[..., tuple[T, int]]:
    """
    Decorator to measure function execution time in milliseconds.
    
    Args:
        func: The function to measure.
        
    Returns:
        A tuple of (function result, latency in ms)
    """
    def wrapper(*args: Any, **kwargs: Any) -> tuple[T, int]:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        return result, latency_ms
    
    return wrapper


class Result(Generic[T]):
    """
    Represents the result of an operation that might fail.
    
    Similar to Rust's Result type, this provides a way to handle
    errors without exceptions.
    """
    
    def __init__(
        self, 
        value: Optional[T] = None, 
        error: Optional[Exception] = None
    ):
        self.value = value
        self.error = error
        
    @property
    def is_ok(self) -> bool:
        """Check if the result is successful."""
        return self.error is None
    
    @property
    def is_err(self) -> bool:
        """Check if the result is an error."""
        return self.error is not None
    
    def unwrap(self) -> T:
        """
        Get the value if successful, otherwise raise the error.
        
        Returns:
            The value if successful.
            
        Raises:
            Exception: The stored error if unsuccessful.
        """
        if self.is_err:
            raise self.error
        return self.value
    
    def unwrap_or(self, default: T) -> T:
        """
        Get the value if successful, otherwise return the default.
        
        Args:
            default: The default value to return if there's an error.
            
        Returns:
            The value if successful, otherwise the default.
        """
        if self.is_err:
            return default
        return self.value
    
    @classmethod
    def ok(cls, value: T) -> 'Result[T]':
        """Create a successful result."""
        return cls(value=value)
    
    @classmethod
    def err(cls, error: Exception) -> 'Result[T]':
        """Create an error result."""
        return cls(error=error)
