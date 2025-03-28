"""
Judge storage implementation using PostgreSQL with pgvector extension.
"""
import os
import uuid
import logging
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple, Union
from sentence_transformers import SentenceTransformer
from sqlalchemy import text, Column, String, Integer, Float, DateTime, JSON, Text, and_, or_, between
from shared.db import Database, Base, Repository
from shared.service_client import ServiceClient
from shared.utils import ApiError

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
        # Safely handle result_metadata to avoid session refresh issues
        # Don't call .copy() which can trigger a session refresh
        if hasattr(self, 'result_metadata') and self.result_metadata is not None:
            try:
                # Try to convert to dict if it's not already
                if isinstance(self.result_metadata, dict):
                    metadata = dict(self.result_metadata)
                else:
                    metadata = dict(self.result_metadata)
            except Exception:
                # Fallback if can't convert
                metadata = None
        else:
            metadata = None
        
        # Handle timestamp similarly
        if hasattr(self, 'timestamp') and self.timestamp is not None:
            try:
                timestamp_str = self.timestamp.isoformat()
            except Exception:
                timestamp_str = None
        else:
            timestamp_str = None
        
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
            "timestamp": timestamp_str,
            "metadata": metadata
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
    ) -> Tuple[List[Dict[str, Any]], int]:
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
            Tuple of (results as dictionaries, total_count)
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
            
            # Convert to dictionaries within the session context
            result_dicts = []
            for result in results:
                # Safely extract all attributes to avoid session dependency
                result_dict = {
                    "id": result.id,
                    "query_id": result.query_id,
                    "query_text": result.query_text,
                    "output_text": result.output_text,
                    "model_id": result.model_id,
                    "theme": result.theme,
                    "evaluation_prompt_id": result.evaluation_prompt_id,
                    "evaluation_prompt": result.evaluation_prompt,
                    "score": float(result.score) if result.score is not None else None,
                    "judge_model": result.judge_model,
                    "timestamp": result.timestamp.isoformat() if result.timestamp else None,
                }
                
                # Handle metadata separately to avoid session refresh issues
                if hasattr(result, 'result_metadata') and result.result_metadata is not None:
                    try:
                        # Make a deep copy of the metadata to detach it from the session
                        if isinstance(result.result_metadata, dict):
                            result_dict["metadata"] = dict(result.result_metadata)
                        else:
                            result_dict["metadata"] = dict(result.result_metadata)
                    except Exception as e:
                        logging.warning(f"Failed to convert metadata to dict: {str(e)}")
                        result_dict["metadata"] = None
                else:
                    result_dict["metadata"] = None
                
                result_dicts.append(result_dict)
            
            return result_dicts, total_count

class PgJudgeStorage:
    """Storage for LLM outputs and evaluation results using PostgreSQL with pgvector extension."""
    
    def __init__(self, postgres_url: str = None):
        """
        Initialize PostgreSQL connections for both LLM outputs and evaluation results.
        
        Args:
            postgres_url: PostgreSQL connection URL
        """
        self.logger = logging.getLogger("pg_judge_storage")
        
        # Setup PostgreSQL connection
        postgres_url = postgres_url or "postgresql://postgres:postgres@postgres:5432/panopticon"
        try:
            self.db = Database(
                connection_string=postgres_url,
                pool_size=10,
                max_overflow=20
            )
            
            # Initialize embedding model for vector searches first
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2
                self.logger.info(f"Initialized embedding model with dimension {self.embedding_dimension}")
            except Exception as e:
                self.logger.error(f"Failed to initialize embedding model: {str(e)}")
                raise
            
            # Create tables after embedding model is initialized
            self._create_tables()
            
            self.logger.info("PostgreSQL connection established")
            
            # Create repository for evaluation results
            self.results_repo = EvaluationResultRepository(self.db, EvaluationResult)
                
        except Exception as e:
            self.logger.error(f"Failed to initialize PostgreSQL: {str(e)}")
            raise
        
        # Initialize client for Model Registry
        model_registry_url = os.environ.get("MODEL_REGISTRY_URL", "http://model-registry:8000")
        self.model_registry_client = ServiceClient(model_registry_url)
        self.logger.info(f"Model Registry client initialized for URL: {model_registry_url}")
        
    def _create_tables(self):
        """Create necessary tables and extensions if they don't exist."""
        try:
            with self.db.engine.connect() as connection:
                # Enable pgvector extension
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                
                # Create llm_outputs table with vector support
                connection.execute(text(f"""
                CREATE TABLE IF NOT EXISTS llm_outputs (
                    id UUID PRIMARY KEY,
                    output_text TEXT NOT NULL,
                    embedding VECTOR({self.embedding_dimension}),
                    model_id TEXT NOT NULL,
                    theme TEXT NOT NULL,
                    query TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
                """))
                
                # Create index for similarity search
                connection.execute(text(f"""
                CREATE INDEX IF NOT EXISTS llm_outputs_embedding_idx
                ON llm_outputs
                USING ivfflat (embedding vector_cosine_ops);
                """))
                
                # Ensure evaluation_results table exists (created by Base.metadata.create_all)
                self.db.create_tables()
                
                connection.commit()
                
                self.logger.info("Judge storage tables and extensions initialized")
        except Exception as e:
            self.logger.error(f"Failed to create tables: {str(e)}")
            raise

    async def run_query_with_llm(
        self, 
        query: str, 
        model_id: str,
        theme: str,
        metadata: Optional[Dict] = None,
        model_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a query through the Model Registry and store the output.
        
        Args:
            query: Query text to run
            model_id: ID of the LLM model to use
            theme: Theme or category of the query
            metadata: Additional metadata
            model_provider: Provider of the LLM model (optional, determined by registry)
            
        Returns:
            Dictionary with output information
        """
        self.logger.info(f"Running query via Model Registry ({model_id}): {query[:50]}...")
        try:
            # Prepare messages for the model
            messages = [{"role": "user", "content": query}]
            
            # Get default temperature from environment
            temperature = float(os.environ.get("MODEL_DEFAULT_TEMPERATURE", "0.7"))
            
            # Call Model Registry Completion Endpoint
            completion_request = {
                "model_id": model_id,
                "messages": messages,
                "temperature": temperature
                # Add max_tokens if needed
            }
            
            completion_response = await self.model_registry_client.post(
                "/api/v1/completion",
                data=completion_request
            )
            
            # Check registry response status
            if completion_response.get("status") != "success" or not completion_response.get("data"):
                error_msg = completion_response.get('message', 'Unknown error from model registry')
                raise ApiError(status_code=500, message=f"Model Registry Error: {error_msg}")
            
            completion_data = completion_response["data"]
            output_text = completion_data.get("content", "")
            output_id = completion_data.get("id", str(uuid.uuid4()))
            provider = completion_data.get("provider_id", "unknown")  # Get provider from registry response
            
            # Generate embedding for the output text
            embedding = self.embedding_model.encode(output_text).tolist()
            
            # Prepare metadata
            output_metadata = {
                "model_id": model_id,
                "theme": theme,
                "query": query,
                "timestamp": datetime.utcnow().isoformat(),
                "provider": provider,  # Use provider from registry
                **(metadata or {})
            }
            
            # Store in PostgreSQL
            with self.db.engine.connect() as connection:
                # Ensure metadata is properly serialized as a JSON string
                serialized_metadata = json.dumps(output_metadata)
                
                connection.execute(
                    text("""
                    INSERT INTO llm_outputs (id, output_text, embedding, model_id, theme, query, metadata, created_at)
                    VALUES (:id, :output_text, :embedding, :model_id, :theme, :query, :metadata, NOW())
                    """),
                    {
                        "id": output_id,
                        "output_text": output_text,
                        "embedding": embedding,
                        "model_id": model_id,
                        "theme": theme,
                        "query": query,
                        "metadata": serialized_metadata
                    }
                )
                connection.commit()
            
            self.logger.info(f"Stored LLM output with ID {output_id}")
            
            return {
                "id": output_id,
                "output": output_text,
                "metadata": output_metadata
            }
        except ApiError as e:
            # Propagate ApiErrors
            self.logger.error(f"API error running query via registry: {e.message}")
            raise e
                
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
        Evaluate an LLM output using a judge LLM via Model Registry.
        
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
        self.logger.info(f"Evaluating output using judge model {judge_model} via Model Registry")
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
            
            # Create message for evaluation
            eval_messages = [{"role": "user", "content": formatted_prompt}]
            
            # Call Model Registry for evaluation
            eval_request = {
                "model_id": judge_model,
                "messages": eval_messages,
                "temperature": 0.2  # Lower temp for deterministic scoring
                # Add max_tokens if needed
            }
            
            eval_response = await self.model_registry_client.post(
                "/api/v1/completion",
                data=eval_request
            )
            
            if eval_response.get("status") != "success" or not eval_response.get("data"):
                error_msg = eval_response.get('message', 'Unknown error from model registry during evaluation')
                raise ApiError(status_code=500, message=f"Model Registry Eval Error: {error_msg}")
            
            eval_data = eval_response["data"]
            score_text = eval_data.get("content", "").strip()
            
            # Try to parse the score, handle potential non-numeric responses
            try:
                score = float(score_text)
                # Clamp score to 1-10 range
                score = max(1.0, min(10.0, score))
            except ValueError:
                self.logger.warning(f"Failed to parse score '{score_text}' from judge {judge_model}")
                # Default to middle score if parsing fails
                score = 5.0
            
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
            with self.db.engine.connect() as connection:
                result = connection.execute(
                    text("""
                    SELECT id, output_text, metadata, created_at
                    FROM llm_outputs
                    WHERE theme = :theme AND model_id = :model_id
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """),
                    {
                        "theme": theme,
                        "model_id": model_id,
                        "limit": limit
                    }
                )
                
                outputs = []
                for row in result:
                    # Ensure metadata is properly deserialized from JSON
                    metadata = row.metadata
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    
                    outputs.append({
                        "id": str(row.id),
                        "output": row.output_text,
                        "metadata": metadata
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
            # Get results and total count from repository
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
            
            # Convert models to dictionaries within the same session context
            # This ensures all attributes are loaded before the session is closed
            result_dicts = []
            for result in results:
                # Safely extract all attributes to avoid session dependency
                result_dict = {
                    "id": result.id,
                    "query_id": result.query_id,
                    "query_text": result.query_text,
                    "output_text": result.output_text,
                    "model_id": result.model_id,
                    "theme": result.theme,
                    "evaluation_prompt_id": result.evaluation_prompt_id,
                    "evaluation_prompt": result.evaluation_prompt,
                    "score": float(result.score) if result.score is not None else None,
                    "judge_model": result.judge_model,
                    "timestamp": result.timestamp.isoformat() if result.timestamp else None,
                }
                
                # Handle metadata separately to avoid session refresh issues
                if hasattr(result, 'result_metadata') and result.result_metadata is not None:
                    try:
                        # Make a deep copy of the metadata to detach it from the session
                        if isinstance(result.result_metadata, dict):
                            result_dict["metadata"] = dict(result.result_metadata)
                        else:
                            result_dict["metadata"] = dict(result.result_metadata)
                    except Exception as e:
                        self.logger.warning(f"Failed to convert metadata to dict: {str(e)}")
                        result_dict["metadata"] = None
                else:
                    result_dict["metadata"] = None
                
                result_dicts.append(result_dict)
            
            return result_dicts, total_count
        except Exception as e:
            self.logger.error(f"Error retrieving evaluation results: {str(e)}")
            raise

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Get available models from the Model Registry.
        
        Returns:
            List of available models with their details
        """
        try:
            response = await self.model_registry_client.get("/api/v1/models")
            if response.get("status") != "success":
                self.logger.error(f"Error getting models from registry: {response.get('message')}")
                raise ApiError(status_code=500, message=f"Failed to get models from registry")
                
            return response.get("data", [])
        except Exception as e:
            self.logger.error(f"Error listing available models: {str(e)}")
            raise ApiError(status_code=500, message=f"Failed to list available models: {str(e)}")

    def search_similar_outputs(self, output_text: str, limit: int = 5) -> List[Dict]:
        """
        Search for similar LLM outputs using semantic similarity.
        
        Args:
            output_text: Output text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of similar outputs with their metadata and distance score
        """
        try:
            # Generate embedding for the output text
            embedding = self.embedding_model.encode(output_text).tolist()
            
            with self.db.engine.connect() as connection:
                # Build the SQL query with the embedding vector directly embedded in the query
                embedding_str = f"[{','.join(str(x) for x in embedding)}]"
                
                # Create the SQL query with the vector data directly in the query string
                query_sql = f"""
                SELECT "id", "output_text", "model_id", "theme", "query", "metadata", 
                       1 - ("embedding" <=> '{embedding_str}'::vector) as similarity
                FROM llm_outputs
                ORDER BY "embedding" <=> '{embedding_str}'::vector
                LIMIT :limit
                """
                
                result = connection.execute(
                    text(query_sql),
                    {
                        "limit": limit
                    }
                )
                
                outputs = []
                for row in result:
                    # Handle metadata properly - it could already be a dict
                    metadata = row.metadata
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                        
                    outputs.append({
                        "id": str(row.id),
                        "output": row.output_text,
                        "model_id": row.model_id,
                        "theme": row.theme,
                        "query": row.query,
                        "metadata": metadata,
                        "distance": 1.0 - float(row.similarity)
                    })
                
                self.logger.info(f"Found {len(outputs)} similar outputs")
                return outputs
        except Exception as e:
            self.logger.error(f"Failed to search similar outputs: {str(e)}")
            raise
