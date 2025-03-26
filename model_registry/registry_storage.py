"""
Storage implementation for model registry.
"""
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, 
    DateTime, JSON, Text, Boolean, ForeignKey, Table
)
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import json
import os
import uuid
from typing import Dict, List, Optional, Any, Tuple
import logging

Base = declarative_base()

class Provider(Base):
    """SQLAlchemy model for LLM providers."""
    __tablename__ = 'providers'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    auth_type = Column(String, default="api_key")
    base_url = Column(String)
    env_var_key = Column(String, nullable=False)
    litellm_provider = Column(String)
    adapter = Column(String, nullable=False)
    supports_streaming = Column(Boolean, default=False)
    config = Column(JSON, default=lambda: {})
    created_at = Column(DateTime, default=datetime.utcnow)

class Model(Base):
    """SQLAlchemy model for LLM models."""
    __tablename__ = 'models'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    provider_id = Column(String, ForeignKey('providers.id'), nullable=False)
    is_judge_compatible = Column(Boolean, default=False)
    capabilities = Column(JSON, nullable=False)
    config = Column(JSON, default=lambda: {})
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to provider
    provider = relationship("Provider", backref="models")

class CompletionLog(Base):
    """SQLAlchemy model for storing completion logs."""
    __tablename__ = 'completion_logs'
    
    id = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey('models.id'), nullable=False)
    query = Column(Text, nullable=False)
    response = Column(Text)
    error = Column(Text)
    completion_metadata = Column(JSON, default=lambda: {})
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to model
    model = relationship("Model", backref="completions")

class ModelRegistryStorage:
    """Storage implementation for model registry."""
    
    def __init__(self, db_url: str = None):
        """
        Initialize storage with database connection.
        
        Args:
            db_url: Database connection URL
        """
        self.logger = logging.getLogger("model_registry_storage")
        
        # Use provided DB URL or construct from environment variables
        if not db_url:
            db_url = os.environ.get(
                "DATABASE_URL", 
                "postgresql://postgres:postgres@postgres:5432/panopticon"
            )
            
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.logger.info(f"Model registry database initialized at {db_url}")
        
        # Load default configurations
        self._load_defaults()
        
    def _load_defaults(self):
        """Load default models and providers from configuration files."""
        try:
            # Load default providers
            providers_path = os.path.join(os.path.dirname(__file__), "config", "default_providers.json")
            if os.path.exists(providers_path):
                with open(providers_path, "r") as f:
                    providers = json.load(f)
                    for provider in providers:
                        self.add_provider(provider, skip_if_exists=True)
                        
            # Load default models
            models_path = os.path.join(os.path.dirname(__file__), "config", "default_models.json")
            if os.path.exists(models_path):
                with open(models_path, "r") as f:
                    models = json.load(f)
                    for model in models:
                        self.add_model(model, skip_if_exists=True)
        except Exception as e:
            self.logger.error(f"Error loading default configurations: {str(e)}")
    
    def get_provider(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """
        Get provider by ID.
        
        Args:
            provider_id: Provider ID
            
        Returns:
            Provider data or None if not found
        """
        with self.Session() as session:
            provider = session.query(Provider).filter(Provider.id == provider_id).first()
            if not provider:
                return None
                
            return {
                "id": provider.id,
                "name": provider.name,
                "auth_type": provider.auth_type,
                "base_url": provider.base_url,
                "env_var_key": provider.env_var_key,
                "litellm_provider": provider.litellm_provider,
                "adapter": provider.adapter,
                "supports_streaming": provider.supports_streaming,
                "config": provider.config,
                "created_at": provider.created_at.isoformat() if provider.created_at else None
            }
    
    def get_all_providers(self) -> List[Dict[str, Any]]:
        """
        Get all providers.
        
        Returns:
            List of provider data
        """
        with self.Session() as session:
            providers = session.query(Provider).all()
            return [
                {
                    "id": provider.id,
                    "name": provider.name,
                    "auth_type": provider.auth_type,
                    "base_url": provider.base_url,
                    "env_var_key": provider.env_var_key,
                    "litellm_provider": provider.litellm_provider,
                    "adapter": provider.adapter,
                    "supports_streaming": provider.supports_streaming,
                    "config": provider.config,
                    "created_at": provider.created_at.isoformat() if provider.created_at else None
                }
                for provider in providers
            ]
    
    def add_provider(self, provider_data: Dict[str, Any], skip_if_exists: bool = False) -> Optional[Dict[str, Any]]:
        """
        Add a new provider to the registry.
        
        Args:
            provider_data: Provider configuration data
            skip_if_exists: Whether to skip if provider already exists
            
        Returns:
            Added provider data or None if skipped
        """
        with self.Session() as session:
            # Check if provider already exists
            existing = session.query(Provider).filter(Provider.id == provider_data["id"]).first()
            if existing and skip_if_exists:
                self.logger.info(f"Provider {provider_data['id']} already exists, skipping")
                return None
                
            if existing:
                # Update existing provider
                for key, value in provider_data.items():
                    if key != "id" and hasattr(existing, key):
                        setattr(existing, key, value)
                session.commit()
                
                self.logger.info(f"Updated provider {provider_data['id']}")
                
                return {
                    "id": existing.id,
                    "name": existing.name,
                    "auth_type": existing.auth_type,
                    "base_url": existing.base_url,
                    "env_var_key": existing.env_var_key,
                    "litellm_provider": existing.litellm_provider,
                    "adapter": existing.adapter,
                    "supports_streaming": existing.supports_streaming,
                    "config": existing.config,
                    "created_at": existing.created_at.isoformat() if existing.created_at else None
                }
            else:
                # Create new provider
                provider = Provider(
                    id=provider_data["id"],
                    name=provider_data["name"],
                    auth_type=provider_data.get("auth_type", "api_key"),
                    base_url=provider_data.get("base_url"),
                    env_var_key=provider_data["env_var_key"],
                    litellm_provider=provider_data.get("litellm_provider"),
                    adapter=provider_data["adapter"],
                    supports_streaming=provider_data.get("supports_streaming", False),
                    config=provider_data.get("config", {}),
                    created_at=datetime.utcnow()
                )
                session.add(provider)
                session.commit()
                
                self.logger.info(f"Added provider {provider_data['id']}")
                
                return {
                    "id": provider.id,
                    "name": provider.name,
                    "auth_type": provider.auth_type,
                    "base_url": provider.base_url,
                    "env_var_key": provider.env_var_key,
                    "litellm_provider": provider.litellm_provider,
                    "adapter": provider.adapter,
                    "supports_streaming": provider.supports_streaming,
                    "config": provider.config,
                    "created_at": provider.created_at.isoformat() if provider.created_at else None
                }
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model by ID.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model data or None if not found
        """
        with self.Session() as session:
            model = session.query(Model).filter(Model.id == model_id).first()
            if not model:
                return None
                
            return {
                "id": model.id,
                "name": model.name,
                "provider_id": model.provider_id,
                "is_judge_compatible": model.is_judge_compatible,
                "capabilities": model.capabilities,
                "config": model.config,
                "created_at": model.created_at.isoformat() if model.created_at else None
            }
    
    def get_all_models(self) -> List[Dict[str, Any]]:
        """
        Get all models.
        
        Returns:
            List of model data
        """
        with self.Session() as session:
            models = session.query(Model).all()
            return [
                {
                    "id": model.id,
                    "name": model.name,
                    "provider_id": model.provider_id,
                    "is_judge_compatible": model.is_judge_compatible,
                    "capabilities": model.capabilities,
                    "config": model.config,
                    "created_at": model.created_at.isoformat() if model.created_at else None
                }
                for model in models
            ]
    
    def add_model(self, model_data: Dict[str, Any], skip_if_exists: bool = False) -> Optional[Dict[str, Any]]:
        """
        Add a new model to the registry.
        
        Args:
            model_data: Model configuration data
            skip_if_exists: Whether to skip if model already exists
            
        Returns:
            Added model data or None if skipped
        """
        with self.Session() as session:
            # Check if model already exists
            existing = session.query(Model).filter(Model.id == model_data["id"]).first()
            if existing and skip_if_exists:
                self.logger.info(f"Model {model_data['id']} already exists, skipping")
                return None
                
            # Check if provider exists
            provider = session.query(Provider).filter(Provider.id == model_data["provider_id"]).first()
            if not provider:
                raise ValueError(f"Provider {model_data['provider_id']} not found")
                
            if existing:
                # Update existing model
                for key, value in model_data.items():
                    if key != "id" and hasattr(existing, key):
                        setattr(existing, key, value)
                session.commit()
                
                self.logger.info(f"Updated model {model_data['id']}")
                
                return {
                    "id": existing.id,
                    "name": existing.name,
                    "provider_id": existing.provider_id,
                    "is_judge_compatible": existing.is_judge_compatible,
                    "capabilities": existing.capabilities,
                    "config": existing.config,
                    "created_at": existing.created_at.isoformat() if existing.created_at else None
                }
            else:
                # Create new model
                model = Model(
                    id=model_data["id"],
                    name=model_data["name"],
                    provider_id=model_data["provider_id"],
                    is_judge_compatible=model_data.get("is_judge_compatible", False),
                    capabilities=model_data["capabilities"],
                    config=model_data.get("config", {}),
                    created_at=datetime.utcnow()
                )
                session.add(model)
                session.commit()
                
                self.logger.info(f"Added model {model_data['id']}")
                
                return {
                    "id": model.id,
                    "name": model.name,
                    "provider_id": model.provider_id,
                    "is_judge_compatible": model.is_judge_compatible,
                    "capabilities": model.capabilities,
                    "config": model.config,
                    "created_at": model.created_at.isoformat() if model.created_at else None
                }
    
    def log_completion(
        self,
        model_id: str,
        query: str,
        response: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log a completion request and response.
        
        Args:
            model_id: Model ID
            query: Query text or JSON string
            response: Response text (if successful)
            error: Error message (if failed)
            metadata: Additional metadata
            
        Returns:
            Logged completion data
        """
        with self.Session() as session:
            # Check if model exists
            model = session.query(Model).filter(Model.id == model_id).first()
            if not model:
                raise ValueError(f"Model {model_id} not found")
                
            # Create completion log
            log_id = str(uuid.uuid4())
            log = CompletionLog(
                id=log_id,
                model_id=model_id,
                query=query,
                response=response,
                error=error,
                completion_metadata=metadata or {},
                created_at=datetime.utcnow()
            )
            session.add(log)
            session.commit()
            
            self.logger.info(f"Logged completion {log_id} for model {model_id}")
            
            return {
                "id": log.id,
                "model_id": log.model_id,
                "query": log.query,
                "response": log.response,
                "error": log.error,
                "metadata": log.completion_metadata,
                "created_at": log.created_at.isoformat() if log.created_at else None
            }
