"""
Core data models for the LLM Evaluation Framework.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class ModelProvider(str, Enum):
    """Supported LLM model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    MISTRAL = "mistral"
    GOOGLE = "google"
    OTHER = "other"


class ThemeCategory(str, Enum):
    """Categories for evaluation themes."""
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


class ModelConfig(BaseModel):
    """Configuration for an LLM model."""
    provider: ModelProvider
    model_id: str
    api_key: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True


class QueryPrompt(BaseModel):
    """A prompt to be sent to an LLM."""
    id: UUID = Field(default_factory=uuid4)
    text: str
    theme: Optional[ThemeCategory] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Explain quantum entanglement to a high school student",
                "theme": "science_technology",
                "metadata": {"complexity": "medium", "target_audience": "high_school"}
            }
        }


class ModelResponse(BaseModel):
    """Response from an LLM model."""
    id: UUID = Field(default_factory=uuid4)
    prompt_id: UUID
    model_config: ModelConfig
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: Optional[int] = None
    tokens_used: Optional[int] = None
    embedding_id: Optional[UUID] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationMetric(str, Enum):
    """Standard evaluation metrics for LLM responses."""
    RELEVANCE = "relevance"
    FACTUAL_ACCURACY = "factual_accuracy"
    COHERENCE = "coherence"
    TOXICITY = "toxicity"
    CREATIVITY = "creativity"
    REASONING = "reasoning"
    INSTRUCTION_FOLLOWING = "instruction_following"
    CORRECTNESS = "correctness"
    CONSISTENCY = "consistency"
    QUALITY = "quality"


class MetricScore(BaseModel):
    """Score for a single evaluation metric."""
    metric: EvaluationMetric
    score: float
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Result of evaluating a model response."""
    id: UUID = Field(default_factory=uuid4)
    response_id: UUID
    run_id: UUID
    evaluator_id: str
    scores: List[MetricScore]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def average_score(self) -> float:
        """Calculate the average score across all metrics."""
        if not self.scores:
            return 0.0
        return sum(score.score for score in self.scores) / len(self.scores)


class EvaluationRun(BaseModel):
    """A complete evaluation cycle for one or more models."""
    id: UUID = Field(default_factory=uuid4)
    models: List[ModelConfig]
    themes: List[ThemeCategory]
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: str = "pending"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchQueryRequest(BaseModel):
    """Request to query multiple LLMs with multiple prompts."""
    models: List[ModelConfig]
    themes: List[ThemeCategory]
    evaluator_ids: List[str]
    metrics: Optional[List[EvaluationMetric]] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchQueryResponse(BaseModel):
    """Response from a batch query."""
    run_id: UUID
    status: str = "pending"
