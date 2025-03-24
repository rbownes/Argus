"""
Judge storage implementation for evaluating LLM outputs.
"""
import chromadb
from chromadb.config import Settings
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple, Union
import uuid
import os
import litellm
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, 
    DateTime, JSON, Text, and_, or_, between
)
from sqlalchemy.sql import func
import logging
from shared.db import Base, Database, Repository

class EvaluationResult(Base):
    """SQLAlchemy model for evaluation results."""
    __tablename__ = 'evaluation_results'
    
    id = Column(String, primary_key=True)
    query_id = Column(String, index=True)
    query_text = Column(Text)
    output_text = Column(Text)
    model_id = Column(String, index=True)
    theme = Column(String, index=True)
    evaluation_prompt_id = Column(String, index=True)
    evaluation_prompt = Column(Text)
    score = Column(Float, index=True)
    judge_model = Column(String, index=True)
    timestamp = Column(DateTime, index=True, default=datetime.utcnow)
    result_metadata = Column(JSON)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "query_id": self.query_id,
            "query_text": self.query_text,
            "output_text": self.output_text,
            "model_id": self.model_id,
            "theme": self.theme,
            "evaluation_prompt_id": self.evaluation_prompt_id,
            "evaluation_prompt": self.evaluation_prompt,
            "score": self.score,
            "judge_model": self.judge_model,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.result_metadata
        }

class EvaluationResultRepository(Repository[EvaluationResult]):
    """Repository for evaluation results."""
    
    def get_filtered_results(
        self,
        theme: Optional[str] = None,
        model_id: Optional[str] = None,
        evaluation_prompt_id: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        skip: int = 0
    ) -> Tuple[List[EvaluationResult], int]:
        """
        Get evaluation results with filtering.
        
        Args:
            theme: Filter by theme
            model_id: Filter by model ID
            evaluation_prompt_id: Filter by evaluation prompt ID
            min_score: Minimum score
            max_score: Maximum score
            start_date: Start date for filtering
            end_date: End date for filtering
            limit: Maximum number of results to return
            skip: Number of results to skip for pagination
            
        Returns:
            Tuple of (results, total_count)
        """
        with self.db.get_session() as session:
            # Build query with filters
            query = session.query(EvaluationResult)
            filter_conditions = []
            
            if theme:
                filter_conditions.append(EvaluationResult.theme == theme)
            if model_id:
                filter_conditions.append(EvaluationResult.model_id == model_id)
            if evaluation_prompt_id:
                filter_conditions.append(EvaluationResult.evaluation_prompt_id == evaluation_prompt_id)
            if min_score is not None:
                filter_conditions.append(EvaluationResult.score >= min_score)
            if max_score is not None:
                filter_conditions.append(EvaluationResult.score <= max_score)
            if start_date and end_date:
                filter_conditions.append(
                    between(EvaluationResult.timestamp, start_date, end_date)
                )
            elif start_date:
                filter_conditions.append(EvaluationResult.timestamp >= start_date)
            elif end_date:
                filter_conditions.append(EvaluationResult.timestamp <= end_date)
            
            if filter_conditions:
                query = query.filter(and_(*filter_conditions))
            
            # Get total count
            total_count = query.count()
            
            # Apply pagination and sort by timestamp descending
            results = query.order_by(EvaluationResult.timestamp.desc()).offset(skip).limit(limit).all()
            
            return results, total_count

class JudgeStorage:
    """Storage for LLM outputs and evaluation results."""
    
    def __init__(
        self, 
        persist_directory: str = "./judge_data",
        postgres_url: str = None
    ):
        """
        Initialize ChromaDB for storing LLM outputs and PostgreSQL for storing evaluation results.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            postgres_url: PostgreSQL connection URL
        """
        self.logger = logging.getLogger("judge_storage")
        
        # Setup ChromaDB for storing LLM outputs
        try:
            self.client = chromadb.Client(Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            ))
            self.collection = self.client.get_or_create_collection(
                name="llm_outputs",
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info(f"ChromaDB collection 'llm_outputs' initialized at {persist_directory}")
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
        
        # Setup PostgreSQL for storing evaluation results
        postgres_url = postgres_url or "postgresql://postgres:postgres@postgres:5432/panopticon"
        try:
            self.db = Database(
                connection_string=postgres_url,
                pool_size=10,
                max_overflow=20
            )
            self.db.create_tables()
            self.logger.info("PostgreSQL connection established")
            
            # Create repository for evaluation results
            self.results_repo = EvaluationResultRepository(self.db, EvaluationResult)
        except Exception as e:
            self.logger.error(f"Failed to initialize PostgreSQL: {str(e)}")
            raise
        
        # Configure LiteLLM
        litellm.verbose = False
        self.logger.info("LiteLLM initialized")

    async def run_query_with_llm(
        self, 
        query: str, 
        model_id: str,
        theme: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run a query through LiteLLM and store the output.
        
        Args:
            query: Query text to run
            model_id: ID of the LLM model to use
            theme: Theme or category of the query
            metadata: Additional metadata
            
        Returns:
            Dictionary with output information
        """
        self.logger.info(f"Running query with model {model_id}: {query[:50]}...")
        try:
            # Run query through LiteLLM
            response = await litellm.acompletion(
                model=model_id,
                messages=[{"role": "user", "content": query}],
                temperature=float(os.environ.get("LITELLM_MODEL_DEFAULT_TEMPERATURE", "0.7"))
            )
            
            output_text = response.choices[0].message.content
            output_id = str(uuid.uuid4())
            
            # Prepare metadata
            output_metadata = {
                "model_id": model_id,
                "theme": theme,
                "query": query,
                "timestamp": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
            
            # Store in ChromaDB
            self.collection.add(
                documents=[output_text],
                metadatas=[output_metadata],
                ids=[output_id]
            )
            
            self.logger.info(f"Stored LLM output with ID {output_id}")
            
            return {
                "id": output_id,
                "output": output_text,
                "metadata": output_metadata
            }
        except Exception as e:
            self.logger.error(f"Error running query with LLM: {str(e)}")
            raise

    async def evaluate_output(
        self,
        query: str,
        output: str,
        evaluation_prompt: str,
        evaluation_prompt_id: str,
        model_id: str,
        theme: str,
        judge_model: str = "gpt-4",
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an LLM output using a judge LLM.
        
        Args:
            query: Original query
            output: LLM output to evaluate
            evaluation_prompt: Prompt for evaluation
            evaluation_prompt_id: ID of the evaluation prompt
            model_id: ID of the LLM model that generated the output
            theme: Theme or category of the query
            judge_model: LLM model to use for evaluation
            metadata: Additional metadata
            
        Returns:
            Dictionary with evaluation result
        """
        self.logger.info(f"Evaluating output using prompt ID {evaluation_prompt_id}")
        try:
            evaluation_prompt_template = """
            You are an expert evaluator. Your task is to evaluate an AI's response based on specific criteria.
            
            Query: {query}
            AI Response: {output}
            
            Evaluation Criteria: {evaluation_prompt}
            
            Please provide a score from 1-10 where:
            1 = Completely fails to meet the criteria
            10 = Perfectly meets the criteria
            
            Respond with ONLY a number between 1 and 10.
            """
            
            formatted_prompt = evaluation_prompt_template.format(
                query=query,
                output=output,
                evaluation_prompt=evaluation_prompt
            )
            
            # Get evaluation from judge LLM
            self.logger.info(f"Using judge model {judge_model}")
            try:
                response = await litellm.acompletion(
                    model=judge_model,
                    messages=[{"role": "user", "content": formatted_prompt}]
                )
                
                # Extract score
                score_text = response.choices[0].message.content.strip()
                
                # Try to parse the score, handle potential non-numeric responses
                try:
                    score = float(score_text)
                    # Clamp score to 1-10 range
                    score = max(1.0, min(10.0, score))
                except ValueError:
                    self.logger.warning(f"Failed to parse score from judge response: {score_text}")
                    # Default to middle score if parsing fails
                    score = 5.0
            except Exception as e:
                self.logger.error(f"Error getting evaluation from judge LLM: {str(e)}")
                raise
            
            # Store evaluation result in PostgreSQL
            result_id = str(uuid.uuid4())
            
            try:
                evaluation_result = self.results_repo.create(
                    id=result_id,
                    query_id=metadata.get("query_id") if metadata else None,
                    query_text=query,
                    output_text=output,
                    model_id=model_id,
                    theme=theme,
                    evaluation_prompt_id=evaluation_prompt_id,
                    evaluation_prompt=evaluation_prompt,
                    score=score,
                    judge_model=judge_model,
                    timestamp=datetime.utcnow(),
                    result_metadata=metadata
                )
                
                self.logger.info(f"Stored evaluation result with ID {result_id}, score: {score}")
                
                return {
                    "id": result_id,
                    "score": score,
                    "metadata": {
                        "query": query,
                        "output": output,
                        "evaluation_prompt": evaluation_prompt,
                        "model_id": model_id,
                        "theme": theme,
                        "judge_model": judge_model,
                        **(metadata or {})
                    }
                }
            except Exception as e:
                self.logger.error(f"Error storing evaluation result: {str(e)}")
                raise
                
        except Exception as e:
            self.logger.error(f"Error in evaluate_output: {str(e)}")
            raise

    def get_outputs_by_theme_and_model(
        self, 
        theme: str, 
        model_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Retrieve LLM outputs by theme and model.
        
        Args:
            theme: Theme to filter by
            model_id: Model ID to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of outputs with their metadata
        """
        try:
            results = self.collection.query(
                query_texts=[""],
                where={"theme": theme, "model_id": model_id},
                n_results=limit
            )
            
            outputs = []
            for i in range(len(results['ids'][0])):
                outputs.append({
                    "id": results['ids'][0][i],
                    "output": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i]
                })
            
            self.logger.info(f"Retrieved {len(outputs)} outputs for theme '{theme}' and model '{model_id}'")
            return outputs
        except Exception as e:
            self.logger.error(f"Error retrieving outputs: {str(e)}")
            raise

    def get_evaluation_results(
        self,
        theme: Optional[str] = None,
        model_id: Optional[str] = None,
        evaluation_prompt_id: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        skip: int = 0
    ) -> Tuple[List[Dict], int]:
        """
        Query evaluation results with filtering.
        
        Args:
            theme: Filter by theme
            model_id: Filter by model ID
            evaluation_prompt_id: Filter by evaluation prompt ID
            min_score: Minimum score
            max_score: Maximum score
            start_date: Start date for filtering
            end_date: End date for filtering
            limit: Maximum number of results to return
            skip: Number of results to skip for pagination
            
        Returns:
            Tuple of (results list, total count)
        """
        try:
            results, total_count = self.results_repo.get_filtered_results(
                theme=theme,
                model_id=model_id,
                evaluation_prompt_id=evaluation_prompt_id,
                min_score=min_score,
                max_score=max_score,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                skip=skip
            )
            
            self.logger.info(f"Retrieved {len(results)} evaluation results (total: {total_count})")
            
            # Convert models to dictionaries
            return [result.to_dict() for result in results], total_count
        except Exception as e:
            self.logger.error(f"Error retrieving evaluation results: {str(e)}")
            raise

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List available models for evaluation.
        
        Returns:
            List of available models with their details
        """
        try:
            # Get models from LiteLLM
            models = await litellm.get_model_list()
            
            # Format model information
            model_info = []
            for model in models:
                model_info.append({
                    "id": model.id,
                    "name": model.display_name if hasattr(model, 'display_name') else model.id,
                    "provider": model.provider if hasattr(model, 'provider') else "unknown",
                    "is_judge_compatible": model.id.startswith("gpt-") or "claude" in model.id
                })
            
            return model_info
        except Exception as e:
            self.logger.error(f"Error listing available models: {str(e)}")
            # Return default models if LiteLLM can't provide the list
            return [
                {"id": "gpt-4", "name": "GPT-4", "provider": "openai", "is_judge_compatible": True},
                {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "provider": "openai", "is_judge_compatible": True},
                {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus", "provider": "anthropic", "is_judge_compatible": True},
                {"id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet", "provider": "anthropic", "is_judge_compatible": True},
                {"id": "gemini-pro", "name": "Gemini Pro", "provider": "google", "is_judge_compatible": False}
            ]
