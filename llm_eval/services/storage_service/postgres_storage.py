"""
PostgreSQL implementation for structured data storage.
"""
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import json

from llm_eval.core.models import (
    Prompt, 
    LLMResponse, 
    EvaluationResult,
    BatchQueryRequest,
    BatchQueryResponse,
    LLMProvider,
    PromptCategory,
    EvaluationType
)
from llm_eval.core.utils import Result, generate_id

# Import asyncpg conditionally to handle environments where it's not installed
try:
    import asyncpg
except ImportError:
    asyncpg = None


class PostgresStorage:
    """
    Database implementation using PostgreSQL.
    
    This handles storing and retrieving structured data.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "postgres",
        password: str = "postgres",
        database: str = "llm_eval"
    ):
        """
        Initialize the PostgreSQL storage.
        
        Args:
            host: Database host.
            port: Database port.
            user: Database user.
            password: Database password.
            database: Database name.
        """
        if asyncpg is None:
            raise ImportError(
                "AsyncPG is not installed. Install it with 'pip install asyncpg'"
            )
        
        self.connection_params = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database
        }
        self.pool = None
    
    async def initialize(self) -> None:
        """Initialize the database connection pool and schema."""
        # Create connection pool
        self.pool = await asyncpg.create_pool(**self.connection_params)
        
        # Create schema if it doesn't exist
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS prompts (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    category TEXT NOT NULL,
                    tags JSONB,
                    metadata JSONB,
                    created_at TIMESTAMP NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS responses (
                    id TEXT PRIMARY KEY,
                    prompt_id TEXT NOT NULL,
                    prompt_text TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    tokens_used INTEGER,
                    latency_ms INTEGER,
                    metadata JSONB,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (prompt_id) REFERENCES prompts(id)
                );
                
                CREATE TABLE IF NOT EXISTS evaluations (
                    id TEXT PRIMARY KEY,
                    response_id TEXT NOT NULL,
                    evaluation_type TEXT NOT NULL,
                    score FLOAT NOT NULL,
                    explanation TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (response_id) REFERENCES responses(id)
                );
                
                CREATE TABLE IF NOT EXISTS batches (
                    id TEXT PRIMARY KEY,
                    request JSONB NOT NULL,
                    response JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL
                );
                
                -- Create indexes
                CREATE INDEX IF NOT EXISTS idx_prompts_category ON prompts(category);
                CREATE INDEX IF NOT EXISTS idx_responses_model_name ON responses(model_name);
                CREATE INDEX IF NOT EXISTS idx_responses_prompt_id ON responses(prompt_id);
                CREATE INDEX IF NOT EXISTS idx_evaluations_response_id ON evaluations(response_id);
                CREATE INDEX IF NOT EXISTS idx_evaluations_type ON evaluations(evaluation_type);
            """)
    
    async def close(self) -> None:
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()
    
    async def store_prompt(self, prompt: Prompt) -> Result[Prompt]:
        """Store a prompt in the database."""
        try:
            if not self.pool:
                await self.initialize()
            
            # Generate an ID if one isn't provided
            if not prompt.id:
                prompt.id = generate_id()
            
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO prompts (id, text, category, tags, metadata, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (id) DO UPDATE
                    SET text = $2, category = $3, tags = $4, metadata = $5
                """, 
                prompt.id,
                prompt.text,
                prompt.category,
                json.dumps(prompt.tags),
                json.dumps(prompt.metadata),
                prompt.created_at
                )
            
            return Result.ok(prompt)
        except Exception as e:
            return Result.err(e)
    
    async def store_response(self, response: LLMResponse) -> Result[LLMResponse]:
        """Store an LLM response in the database."""
        try:
            if not self.pool:
                await self.initialize()
            
            # Generate an ID if one isn't provided
            if not response.id:
                response.id = generate_id()
            
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO responses (
                        id, prompt_id, prompt_text, model_name, provider,
                        response_text, tokens_used, latency_ms, metadata, created_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (id) DO UPDATE
                    SET prompt_id = $2, prompt_text = $3, model_name = $4, provider = $5,
                        response_text = $6, tokens_used = $7, latency_ms = $8, metadata = $9
                """, 
                response.id,
                response.prompt_id,
                response.prompt_text,
                response.model_name,
                response.provider,
                response.response_text,
                response.tokens_used,
                response.latency_ms,
                json.dumps(response.metadata),
                response.created_at
                )
            
            return Result.ok(response)
        except Exception as e:
            return Result.err(e)
    
    async def store_evaluation(self, evaluation: EvaluationResult) -> Result[EvaluationResult]:
        """Store an evaluation result in the database."""
        try:
            if not self.pool:
                await self.initialize()
            
            # Generate an ID if one isn't provided
            if not evaluation.id:
                evaluation.id = generate_id()
            
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO evaluations (
                        id, response_id, evaluation_type, score,
                        explanation, metadata, created_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (id) DO UPDATE
                    SET response_id = $2, evaluation_type = $3, score = $4,
                        explanation = $5, metadata = $6
                """, 
                evaluation.id,
                evaluation.response_id,
                evaluation.evaluation_type,
                evaluation.score,
                evaluation.explanation,
                json.dumps(evaluation.metadata),
                evaluation.created_at
                )
            
            return Result.ok(evaluation)
        except Exception as e:
            return Result.err(e)
    
    async def store_batch(
        self, 
        batch_request: BatchQueryRequest,
        batch_response: BatchQueryResponse
    ) -> Result[str]:
        """Store a batch query request and response."""
        try:
            if not self.pool:
                await self.initialize()
            
            batch_id = batch_response.batch_id
            
            async with self.pool.acquire() as conn:
                # Store the batch
                await conn.execute("""
                    INSERT INTO batches (id, request, response, created_at)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (id) DO UPDATE
                    SET request = $2, response = $3
                """, 
                batch_id,
                json.dumps(batch_request.dict()),
                json.dumps(batch_response.dict()),
                datetime.now()
                )
            
            # Also store individual responses
            for response in batch_response.responses:
                await self.store_response(response)
            
            # Store evaluations if present
            if batch_response.evaluations:
                for evaluation in batch_response.evaluations:
                    await self.store_evaluation(evaluation)
            
            return Result.ok(batch_id)
        except Exception as e:
            return Result.err(e)
    
    async def get_prompt(self, prompt_id: str) -> Result[Prompt]:
        """Get a prompt by ID."""
        try:
            if not self.pool:
                await self.initialize()
            
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT id, text, category, tags, metadata, created_at
                    FROM prompts
                    WHERE id = $1
                """, prompt_id)
                
                if not row:
                    return Result.err(KeyError(f"Prompt with ID {prompt_id} not found"))
                
                prompt = Prompt(
                    id=row['id'],
                    text=row['text'],
                    category=PromptCategory(row['category']),
                    tags=json.loads(row['tags']),
                    metadata=json.loads(row['metadata']),
                    created_at=row['created_at']
                )
                
                return Result.ok(prompt)
        except Exception as e:
            return Result.err(e)
    
    async def get_response(self, response_id: str) -> Result[LLMResponse]:
        """Get an LLM response by ID."""
        try:
            if not self.pool:
                await self.initialize()
            
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT id, prompt_id, prompt_text, model_name, provider,
                           response_text, tokens_used, latency_ms, metadata, created_at
                    FROM responses
                    WHERE id = $1
                """, response_id)
                
                if not row:
                    return Result.err(KeyError(f"Response with ID {response_id} not found"))
                
                response = LLMResponse(
                    id=row['id'],
                    prompt_id=row['prompt_id'],
                    prompt_text=row['prompt_text'],
                    model_name=row['model_name'],
                    provider=LLMProvider(row['provider']),
                    response_text=row['response_text'],
                    tokens_used=row['tokens_used'],
                    latency_ms=row['latency_ms'],
                    metadata=json.loads(row['metadata']),
                    created_at=row['created_at']
                )
                
                return Result.ok(response)
        except Exception as e:
            return Result.err(e)
    
    async def get_evaluation(self, evaluation_id: str) -> Result[EvaluationResult]:
        """Get an evaluation result by ID."""
        try:
            if not self.pool:
                await self.initialize()
            
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT id, response_id, evaluation_type, score,
                           explanation, metadata, created_at
                    FROM evaluations
                    WHERE id = $1
                """, evaluation_id)
                
                if not row:
                    return Result.err(KeyError(f"Evaluation with ID {evaluation_id} not found"))
                
                evaluation = EvaluationResult(
                    id=row['id'],
                    response_id=row['response_id'],
                    evaluation_type=EvaluationType(row['evaluation_type']),
                    score=row['score'],
                    explanation=row['explanation'],
                    metadata=json.loads(row['metadata']),
                    created_at=row['created_at']
                )
                
                return Result.ok(evaluation)
        except Exception as e:
            return Result.err(e)
    
    async def get_batch(self, batch_id: str) -> Result[BatchQueryResponse]:
        """Get a batch query response by ID."""
        try:
            if not self.pool:
                await self.initialize()
            
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT id, request, response
                    FROM batches
                    WHERE id = $1
                """, batch_id)
                
                if not row:
                    return Result.err(KeyError(f"Batch with ID {batch_id} not found"))
                
                batch_response_data = json.loads(row['response'])
                batch_response = BatchQueryResponse(**batch_response_data)
                
                return Result.ok(batch_response)
        except Exception as e:
            return Result.err(e)
    
    async def query_responses(
        self,
        model_name: Optional[str] = None,
        prompt_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Result[List[LLMResponse]]:
        """Query LLM responses with filters."""
        try:
            if not self.pool:
                await self.initialize()
            
            # Build query conditions
            conditions = []
            params = []
            
            if model_name:
                conditions.append(f"model_name = ${len(params) + 1}")
                params.append(model_name)
            
            if prompt_id:
                conditions.append(f"prompt_id = ${len(params) + 1}")
                params.append(prompt_id)
            
            if start_time:
                conditions.append(f"created_at >= ${len(params) + 1}")
                params.append(start_time)
            
            if end_time:
                conditions.append(f"created_at <= ${len(params) + 1}")
                params.append(end_time)
            
            # Build the query
            query = """
                SELECT id, prompt_id, prompt_text, model_name, provider,
                       response_text, tokens_used, latency_ms, metadata, created_at
                FROM responses
            """
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += f" ORDER BY created_at DESC LIMIT {limit} OFFSET {offset}"
            
            # Execute the query
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                responses = []
                for row in rows:
                    response = LLMResponse(
                        id=row['id'],
                        prompt_id=row['prompt_id'],
                        prompt_text=row['prompt_text'],
                        model_name=row['model_name'],
                        provider=LLMProvider(row['provider']),
                        response_text=row['response_text'],
                        tokens_used=row['tokens_used'],
                        latency_ms=row['latency_ms'],
                        metadata=json.loads(row['metadata']),
                        created_at=row['created_at']
                    )
                    responses.append(response)
                
                return Result.ok(responses)
        except Exception as e:
            return Result.err(e)
    
    async def query_evaluations(
        self,
        response_id: Optional[str] = None,
        evaluation_type: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Result[List[EvaluationResult]]:
        """Query evaluation results with filters."""
        try:
            if not self.pool:
                await self.initialize()
            
            # Build query conditions
            conditions = []
            params = []
            
            if response_id:
                conditions.append(f"response_id = ${len(params) + 1}")
                params.append(response_id)
            
            if evaluation_type:
                conditions.append(f"evaluation_type = ${len(params) + 1}")
                params.append(evaluation_type)
            
            if min_score is not None:
                conditions.append(f"score >= ${len(params) + 1}")
                params.append(min_score)
            
            if max_score is not None:
                conditions.append(f"score <= ${len(params) + 1}")
                params.append(max_score)
            
            # Build the query
            query = """
                SELECT id, response_id, evaluation_type, score,
                       explanation, metadata, created_at
                FROM evaluations
            """
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += f" ORDER BY score DESC LIMIT {limit} OFFSET {offset}"
            
            # Execute the query
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                evaluations = []
                for row in rows:
                    evaluation = EvaluationResult(
                        id=row['id'],
                        response_id=row['response_id'],
                        evaluation_type=EvaluationType(row['evaluation_type']),
                        score=row['score'],
                        explanation=row['explanation'],
                        metadata=json.loads(row['metadata']),
                        created_at=row['created_at']
                    )
                    evaluations.append(evaluation)
                
                return Result.ok(evaluations)
        except Exception as e:
            return Result.err(e)
