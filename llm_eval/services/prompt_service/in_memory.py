"""
In-memory implementation of the Prompt Service.
"""
import json
import os
from typing import Dict, List, Optional, Set

from llm_eval.core.models import Prompt, PromptCategory
from llm_eval.core.utils import Result, generate_id
from .interface import PromptServiceInterface


class InMemoryPromptService(PromptServiceInterface):
    """
    In-memory implementation of the Prompt Service.
    
    This implementation stores all prompts in memory and is intended
    for testing or small-scale deployments.
    """
    
    def __init__(self):
        """Initialize the service with an empty prompt store."""
        self._prompts: Dict[str, Prompt] = {}
    
    async def create_prompt(self, prompt: Prompt) -> Result[Prompt]:
        """Create a new prompt."""
        try:
            # Generate an ID if one isn't provided
            if not prompt.id:
                prompt.id = generate_id()
            
            # Store the prompt
            self._prompts[prompt.id] = prompt
            return Result.ok(prompt)
        except Exception as e:
            return Result.err(e)
    
    async def get_prompt(self, prompt_id: str) -> Result[Prompt]:
        """Get a prompt by ID."""
        try:
            if prompt_id not in self._prompts:
                return Result.err(KeyError(f"Prompt with ID {prompt_id} not found"))
            return Result.ok(self._prompts[prompt_id])
        except Exception as e:
            return Result.err(e)
    
    async def list_prompts(
        self,
        category: Optional[PromptCategory] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Result[List[Prompt]]:
        """List prompts with optional filtering."""
        try:
            prompts = list(self._prompts.values())
            
            # Filter by category if provided
            if category:
                prompts = [p for p in prompts if p.category == category]
            
            # Filter by tags if provided
            if tags:
                tag_set = set(tags)
                prompts = [p for p in prompts if set(p.tags).intersection(tag_set)]
            
            # Apply pagination
            paginated = prompts[offset:offset + limit]
            
            return Result.ok(paginated)
        except Exception as e:
            return Result.err(e)
    
    async def update_prompt(self, prompt_id: str, prompt: Prompt) -> Result[Prompt]:
        """Update an existing prompt."""
        try:
            if prompt_id not in self._prompts:
                return Result.err(KeyError(f"Prompt with ID {prompt_id} not found"))
            
            # Ensure ID is preserved
            prompt.id = prompt_id
            
            # Update the prompt
            self._prompts[prompt_id] = prompt
            return Result.ok(prompt)
        except Exception as e:
            return Result.err(e)
    
    async def delete_prompt(self, prompt_id: str) -> Result[bool]:
        """Delete a prompt."""
        try:
            if prompt_id not in self._prompts:
                return Result.err(KeyError(f"Prompt with ID {prompt_id} not found"))
            
            # Remove the prompt
            del self._prompts[prompt_id]
            return Result.ok(True)
        except Exception as e:
            return Result.err(e)
    
    async def import_prompts_from_file(self, file_path: str) -> Result[List[Prompt]]:
        """Import prompts from a file."""
        try:
            if not os.path.exists(file_path):
                return Result.err(FileNotFoundError(f"File {file_path} not found"))
            
            imported_prompts = []
            
            # Handle different file types
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Handle different JSON formats
                if isinstance(data, list):
                    # List of prompts
                    for item in data:
                        prompt = Prompt(**item)
                        result = await self.create_prompt(prompt)
                        if result.is_ok:
                            imported_prompts.append(result.unwrap())
                elif isinstance(data, dict) and 'prompts' in data:
                    # Dict with prompts key
                    for item in data['prompts']:
                        prompt = Prompt(**item)
                        result = await self.create_prompt(prompt)
                        if result.is_ok:
                            imported_prompts.append(result.unwrap())
                elif isinstance(data, dict):
                    # Dict of theme -> prompts
                    for theme, prompts in data.items():
                        for prompt_text in prompts:
                            prompt = Prompt(
                                text=prompt_text,
                                category=theme,
                                tags=[theme]
                            )
                            result = await self.create_prompt(prompt)
                            if result.is_ok:
                                imported_prompts.append(result.unwrap())
            elif file_path.endswith('.py'):
                # Handle Python file (assumes it's similar to diverse_queries.py)
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Very simple parsing - this is not robust
                if 'DIVERSE_QUERIES' in content:
                    # Extract as Python code and evaluate
                    # Warning: This is unsafe for arbitrary files
                    namespace = {}
                    exec(content, namespace)
                    
                    if 'DIVERSE_QUERIES' in namespace:
                        queries = namespace['DIVERSE_QUERIES']
                        for theme, prompts in queries.items():
                            for prompt_text in prompts:
                                prompt = Prompt(
                                    text=prompt_text,
                                    category=theme,
                                    tags=[theme]
                                )
                                result = await self.create_prompt(prompt)
                                if result.is_ok:
                                    imported_prompts.append(result.unwrap())
            
            return Result.ok(imported_prompts)
        except Exception as e:
            return Result.err(e)
    
    async def search_prompts(self, query: str) -> Result[List[Prompt]]:
        """Search prompts by text."""
        try:
            query = query.lower()
            matching_prompts = []
            
            for prompt in self._prompts.values():
                # Simple substring search
                if query in prompt.text.lower():
                    matching_prompts.append(prompt)
                # Also search in tags
                elif any(query in tag.lower() for tag in prompt.tags):
                    matching_prompts.append(prompt)
            
            return Result.ok(matching_prompts)
        except Exception as e:
            return Result.err(e)
