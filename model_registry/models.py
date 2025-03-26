"""
Pydantic models for the Model Registry service.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime

class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    id: str = Field(..., description="Provider ID (e.g., 'openai', 'anthropic', 'gemini')")
    name: str = Field(..., description="Provider display name")
    auth_type: Literal["api_key", "oauth"] = Field("api_key", description="Authentication type")
    base_url: Optional[str] = Field(None, description="Base URL for API calls")
    env_var_key: str = Field(..., description="Environment variable name for API key")
    litellm_provider: Optional[str] = Field(None, description="LiteLLM provider name (if applicable)")
    adapter: str = Field(..., description="Adapter class to use")
    supports_streaming: bool = Field(False, description="Whether provider supports streaming")
    
class ModelConfig(BaseModel):
    """Configuration for an LLM model."""
    id: str = Field(..., description="Model ID")
    name: str = Field(..., description="Model display name")
    provider_id: str = Field(..., description="ID of the provider")
    is_judge_compatible: bool = Field(False, description="Whether the model can be used as a judge")
    capabilities: List[Literal["completion", "chat", "embedding"]] = Field(..., description="Model capabilities")
    config: Dict[str, Any] = Field({}, description="Model-specific configuration")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
class CompletionRequest(BaseModel):
    """Unified completion request."""
    model_id: str = Field(..., description="Model ID to use")
    messages: List[Dict[str, str]] = Field(..., description="Messages for chat completion")
    temperature: Optional[float] = Field(None, description="Temperature parameter")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for completion")
    
class CompletionResponse(BaseModel):
    """Unified completion response."""
    id: str = Field(..., description="Response ID")
    model_id: str = Field(..., description="Model ID used")
    provider_id: str = Field(..., description="Provider ID used")
    content: str = Field(..., description="Generated content")
    usage: Dict[str, int] = Field({}, description="Token usage statistics")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
