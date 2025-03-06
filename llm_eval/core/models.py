"""
Core data models for the LLM Evaluation Framework.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field


class PromptCategory(str, Enum):
    """Categories for organizing prompts."""
    SCIENCE_TECHNOLOGY = "science_technology"
    ARTS_LITERATURE = "arts_literature"
    HISTORY_CULTURE = "history_culture"
    PHILOSOPHY_ETHICS = "philosophy_ethics"
    BUSINESS_ECONOMICS = "business_economics"
    HEALTH_MEDICINE = "health_medicine"
    ENVIRONMENT_SUSTAINABILITY = "environment_sustainability"
    PERSONAL_DEVELOPMENT = "personal_development"
    SOCIAL_ISSUES_POLITICS = "social_issues_politics"
    MATHEMATICS_LOGIC = "mathematics_logic"
    OTHER = "other"


class Prompt(BaseModel):
    """A prompt for querying LLMs."""
    id: Optional[str] = None
    text: str
    category: PromptCategory = PromptCategory.OTHER
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Explain quantum entanglement to a high school student",
                "category": "science_technology",
                "tags": ["physics", "quantum", "educational"]
            }
        }


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    MISTRAL = "mistral"
    GOOGLE = "google"
    OTHER = "other"


class LLMResponse(BaseModel):
    """A response from an LLM."""
    id: Optional[str] = None
    prompt_id: str
    prompt_text: str
    model_name: str
    provider: LLMProvider
    response_text: str
    tokens_used: Optional[int] = None
    latency_ms: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class EvaluationType(str, Enum):
    """Types of evaluations that can be performed."""
    TOXICITY = "toxicity"
    QUALITY = "quality"
    FACTUALITY = "factuality"
    CONSISTENCY = "consistency"
    RELEVANCE = "relevance"
    CORRECTNESS = "correctness"
    COHERENCE = "coherence"
    CUSTOM = "custom"


class EvaluationResult(BaseModel):
    """Result of an evaluation on an LLM response."""
    id: Optional[str] = None
    response_id: str
    evaluation_type: EvaluationType
    score: float
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class BatchQueryRequest(BaseModel):
    """Request to query multiple LLMs with multiple prompts."""
    prompt_ids: List[str]
    model_names: List[str]
    evaluations: List[EvaluationType] = []
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchQueryResponse(BaseModel):
    """Response from a batch query."""
    batch_id: str
    responses: List[LLMResponse]
    evaluations: Optional[List[EvaluationResult]] = None
