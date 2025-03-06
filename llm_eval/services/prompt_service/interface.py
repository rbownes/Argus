"""
Interface for the Prompt Service.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from llm_eval.core.models import Prompt, PromptCategory
from llm_eval.core.utils import Result


class PromptServiceInterface(ABC):
    """Interface for services that manage prompts."""
    
    @abstractmethod
    async def create_prompt(self, prompt: Prompt) -> Result[Prompt]:
        """
        Create a new prompt.
        
        Args:
            prompt: The prompt to create.
            
        Returns:
            Result containing the created prompt with its ID.
        """
        pass
    
    @abstractmethod
    async def get_prompt(self, prompt_id: str) -> Result[Prompt]:
        """
        Get a prompt by ID.
        
        Args:
            prompt_id: The ID of the prompt to retrieve.
            
        Returns:
            Result containing the prompt if found.
        """
        pass
    
    @abstractmethod
    async def list_prompts(
        self, 
        category: Optional[PromptCategory] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Result[List[Prompt]]:
        """
        List prompts with optional filtering.
        
        Args:
            category: Filter by category if provided.
            tags: Filter by tags if provided.
            limit: Maximum number of prompts to return.
            offset: Number of prompts to skip.
            
        Returns:
            Result containing a list of prompts.
        """
        pass
    
    @abstractmethod
    async def update_prompt(self, prompt_id: str, prompt: Prompt) -> Result[Prompt]:
        """
        Update an existing prompt.
        
        Args:
            prompt_id: The ID of the prompt to update.
            prompt: The updated prompt data.
            
        Returns:
            Result containing the updated prompt.
        """
        pass
    
    @abstractmethod
    async def delete_prompt(self, prompt_id: str) -> Result[bool]:
        """
        Delete a prompt.
        
        Args:
            prompt_id: The ID of the prompt to delete.
            
        Returns:
            Result containing True if successful.
        """
        pass
    
    @abstractmethod
    async def import_prompts_from_file(self, file_path: str) -> Result[List[Prompt]]:
        """
        Import prompts from a file.
        
        Args:
            file_path: Path to the file containing prompts.
            
        Returns:
            Result containing a list of imported prompts.
        """
        pass
    
    @abstractmethod
    async def search_prompts(self, query: str) -> Result[List[Prompt]]:
        """
        Search prompts by text.
        
        Args:
            query: Search query.
            
        Returns:
            Result containing a list of matching prompts.
        """
        pass
