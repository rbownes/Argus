"""
Configuration management for all microservices in the Panopticon system.
"""
from typing import Dict, Optional, Any
import os
from pydantic import BaseModel, Field, validator

class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    host: str = Field(default="postgres")
    port: int = Field(default=5432)
    username: str = Field(default="postgres")
    password: str = Field(default="postgres")
    database: str = Field(default="panopticon")
    
    def get_connection_string(self) -> str:
        """Get SQLAlchemy connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class LLMConfig(BaseModel):
    """Language model configuration."""
    litellm_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    default_temperature: float = Field(default=0.7)
    
    @validator('default_temperature')
    def validate_temperature(cls, v):
        """Validate temperature is between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v

class ServiceConfig(BaseModel):
    """Base configuration for all services."""
    service_name: str
    log_level: str = Field(default="INFO")
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    
    @classmethod
    def from_env(cls, service_name: str) -> 'ServiceConfig':
        """Create configuration from environment variables."""
        # Database config
        db_config = DatabaseConfig(
            host=os.environ.get("POSTGRES_HOST", "postgres"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            username=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
            database=os.environ.get("POSTGRES_DB", "panopticon")
        )
        
        # LLM config
        llm_config = LLMConfig(
            litellm_api_key=os.environ.get("LITELLM_API_KEY"),
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            default_temperature=float(os.environ.get("LITELLM_MODEL_DEFAULT_TEMPERATURE", "0.7"))
        )
        
        return cls(
            service_name=service_name,
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            database=db_config,
            llm=llm_config
        )
