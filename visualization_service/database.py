"""
Database utilities for the visualization service.
"""
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import logging
from contextlib import contextmanager, asynccontextmanager
from sqlalchemy.orm import sessionmaker
import json
import asyncio

logger = logging.getLogger("visualization_service.database")

class VisualizationDB:
    """Database connection and query methods for visualization data."""
    
    def __init__(self, connection_string: str, pool_size: int = 5, max_overflow: int = 10):
        """
        Initialize database connection.
        
        Args:
            connection_string: SQLAlchemy connection string
            pool_size: Connection pool size
            max_overflow: Maximum number of connections to overflow
        """
        # Use create_async_engine and specify async driver (e.g., asyncpg)
        # Ensure connection_string starts with postgresql+asyncpg://
        async_connection_string = connection_string.replace("postgresql://", "postgresql+asyncpg://", 1)
        self.engine = create_async_engine(
            async_connection_string,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True  # Check connection before using from pool
        )
        # Use AsyncSession
        self.async_session_factory = sessionmaker(
            bind=self.engine, class_=AsyncSession, expire_on_commit=False
        )
        self.logger = logging.getLogger("visualization_service.database.async")
    
    @contextmanager
    def get_session(self):
        """
        Get a database session with automatic cleanup.
        Note: This is kept for backward compatibility but should be replaced with get_async_session.
        """
        self.logger.warning("Using synchronous get_session() is deprecated. Use get_async_session() instead.")
        raise NotImplementedError("Synchronous sessions are no longer supported. Use get_async_session instead.")
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncSession:
        """Get an async database session with automatic cleanup."""
        session: AsyncSession = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            self.logger.exception("Async Database error: %s", str(e))
            raise
        finally:
            await session.close()
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a raw SQL query asynchronously.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of dictionaries with query results
        """
        async with self.engine.connect() as connection:
            result = await connection.execute(text(query), params or {})
            # More robust row processing with explicit column names
            rows = []
            async for row in result:
                row_dict = {}
                for column, value in row._mapping.items():
                    # Handle potential JSON decoding if needed here
                    if isinstance(value, str) and column == 'result_metadata':
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            pass
                    row_dict[str(column)] = value
                rows.append(row_dict)
            return rows
    
    async def check_connection(self) -> bool:
        """Check if database connection is working."""
        try:
            async with self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            return True
        except Exception as e:
            self.logger.error("Async Database connection check failed: %s", str(e))
            return False
    
    async def get_model_performance_over_time(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        models: Optional[List[str]] = None,
        themes: Optional[List[str]] = None,
        time_grouping: str = "day",
        include_metadata: bool = False
    ) -> pd.DataFrame:
        """
        Get model performance metrics over time.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            models: List of model IDs to include
            themes: List of themes to include
            time_grouping: Time grouping (day, week, month)
            include_metadata: Whether to include result metadata
            
        Returns:
            DataFrame with model performance metrics
        """
        # Default date range if not specified (last 30 days)
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Build query with filters
        query = """
        SELECT 
            model_id,
            theme,
            date_trunc(:time_grouping, timestamp) as time_period,
            AVG(score) as avg_score,
            COUNT(*) as eval_count,
            MIN(score) as min_score,
            MAX(score) as max_score,
            STDDEV(score) as stddev_score
        FROM 
            evaluation_results
        WHERE 
            timestamp BETWEEN :start_date AND :end_date
        """
        
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "time_grouping": time_grouping
        }
        
        # Add model filter if specified
        if models:
            placeholders = [f":model_{i}" for i in range(len(models))]
            model_params = {f"model_{i}": model for i, model in enumerate(models)}
            query += f" AND model_id IN ({', '.join(placeholders)})"
            params.update(model_params)
        
        # Add theme filter if specified
        if themes:
            placeholders = [f":theme_{i}" for i in range(len(themes))]
            theme_params = {f"theme_{i}": theme for i, theme in enumerate(themes)}
            query += f" AND theme IN ({', '.join(placeholders)})"
            params.update(theme_params)
        
        # Group by and order by
        query += """
        GROUP BY 
            model_id, theme, time_period
        ORDER BY 
            time_period, model_id, theme
        """
        
        # Execute query
        try:
            results = await self.execute_query(query, params)
            if not results:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Convert to datetime if time_period exists
            if 'time_period' in df.columns:
                df['time_period'] = pd.to_datetime(df['time_period'])
            
            return df
        except Exception as e:
            self.logger.error(f"Error getting model performance over time: {str(e)}")
            return pd.DataFrame()
    
    async def get_model_comparison(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        models: Optional[List[str]] = None,
        themes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get model comparison data.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            models: List of model IDs to include
            themes: List of themes to include
            
        Returns:
            DataFrame with model comparison data
        """
        # Default date range if not specified (last 30 days)
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Build query with filters
        query = """
        SELECT 
            model_id,
            theme,
            COUNT(*) as eval_count,
            AVG(score) as avg_score,
            MIN(score) as min_score,
            MAX(score) as max_score,
            STDDEV(score) as stddev_score,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY score) as median_score
        FROM 
            evaluation_results
        WHERE 
            timestamp BETWEEN :start_date AND :end_date
        """
        
        params = {
            "start_date": start_date,
            "end_date": end_date
        }
        
        # Add model filter if specified
        if models:
            placeholders = [f":model_{i}" for i in range(len(models))]
            model_params = {f"model_{i}": model for i, model in enumerate(models)}
            query += f" AND model_id IN ({', '.join(placeholders)})"
            params.update(model_params)
        
        # Add theme filter if specified
        if themes:
            placeholders = [f":theme_{i}" for i in range(len(themes))]
            theme_params = {f"theme_{i}": theme for i, theme in enumerate(themes)}
            query += f" AND theme IN ({', '.join(placeholders)})"
            params.update(theme_params)
        
        # Group by and order by
        query += """
        GROUP BY 
            model_id, theme
        ORDER BY 
            avg_score DESC, model_id, theme
        """
        
        # Execute query
        try:
            results = await self.execute_query(query, params)
            if not results:
                return pd.DataFrame()
            
            return pd.DataFrame(results)
        except Exception as e:
            self.logger.error(f"Error getting model comparison: {str(e)}")
            return pd.DataFrame()
    
    async def get_theme_analysis(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        models: Optional[List[str]] = None,
        themes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get theme analysis data for heatmap.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            models: List of model IDs to include
            themes: List of themes to include
            
        Returns:
            DataFrame with theme analysis data
        """
        # Get model comparison data
        df = await self.get_model_comparison(start_date, end_date, models, themes)
        
        if df.empty:
            return pd.DataFrame()
        
        # Pivot for heatmap (model_id vs theme)
        try:
            heatmap_df = df.pivot(index='model_id', columns='theme', values='avg_score')
            return heatmap_df
        except Exception as e:
            self.logger.error(f"Error creating theme analysis heatmap: {str(e)}")
            return pd.DataFrame()
    
    async def get_score_distribution(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        models: Optional[List[str]] = None,
        themes: Optional[List[str]] = None,
        bins: int = 10
    ) -> Dict[str, Any]:
        """
        Get score distribution data for histograms.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            models: List of model IDs to include
            themes: List of themes to include
            bins: Number of bins for histogram
            
        Returns:
            Dictionary with score distribution data
        """
        # Default date range if not specified (last 30 days)
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Build query with filters
        query = """
        SELECT 
            model_id,
            score
        FROM 
            evaluation_results
        WHERE 
            timestamp BETWEEN :start_date AND :end_date
        """
        
        params = {
            "start_date": start_date,
            "end_date": end_date
        }
        
        # Add model filter if specified
        if models:
            placeholders = [f":model_{i}" for i in range(len(models))]
            model_params = {f"model_{i}": model for i, model in enumerate(models)}
            query += f" AND model_id IN ({', '.join(placeholders)})"
            params.update(model_params)
        
        # Add theme filter if specified
        if themes:
            placeholders = [f":theme_{i}" for i in range(len(themes))]
            theme_params = {f"theme_{i}": theme for i, theme in enumerate(themes)}
            query += f" AND theme IN ({', '.join(placeholders)})"
            params.update(theme_params)
        
        # Order by
        query += """
        ORDER BY 
            model_id
        """
        
        # Execute query
        try:
            results = await self.execute_query(query, params)
            if not results:
                return {}
            
            df = pd.DataFrame(results)
            
            # Calculate histograms by model
            histogram_data = {}
            for model in df['model_id'].unique():
                model_scores = df[df['model_id'] == model]['score']
                hist, bin_edges = pd.cut(model_scores, bins=bins, retbins=True)
                hist_counts = hist.value_counts().sort_index()
                
                histogram_data[model] = {
                    'counts': hist_counts.tolist(),
                    'bin_edges': bin_edges.tolist()
                }
            
            return histogram_data
        except Exception as e:
            self.logger.error(f"Error getting score distribution: {str(e)}")
            return {}
    
    async def get_filtered_results(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        models: Optional[List[str]] = None,
        themes: Optional[List[str]] = None,
        evaluation_prompt_ids: Optional[List[str]] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "timestamp",
        sort_desc: bool = True
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get filtered evaluation results with pagination.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            models: List of model IDs to include
            themes: List of themes to include
            evaluation_prompt_ids: List of evaluation prompt IDs to include
            min_score: Minimum score
            max_score: Maximum score
            limit: Maximum number of results to return
            offset: Number of results to skip for pagination
            sort_by: Column to sort by
            sort_desc: Whether to sort in descending order
            
        Returns:
            Tuple of (results, total_count)
        """
        # Default date range if not specified (last 30 days)
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Validate sort_by
        valid_sort_columns = [
            "timestamp", "score", "model_id", "theme", "evaluation_prompt_id"
        ]
        if sort_by not in valid_sort_columns:
            sort_by = "timestamp"
        
        # Build query with filters
        count_query = "SELECT COUNT(*) as total FROM evaluation_results WHERE "
        data_query = """
        SELECT 
            id,
            query_id,
            query_text,
            output_text,
            model_id,
            theme,
            evaluation_prompt_id,
            evaluation_prompt,
            score,
            judge_model,
            timestamp,
            result_metadata
        FROM 
            evaluation_results
        WHERE 
        """
        
        where_clause = "timestamp BETWEEN :start_date AND :end_date"
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
            "offset": offset
        }
        
        # Add model filter if specified
        if models:
            placeholders = [f":model_{i}" for i in range(len(models))]
            model_params = {f"model_{i}": model for i, model in enumerate(models)}
            where_clause += f" AND model_id IN ({', '.join(placeholders)})"
            params.update(model_params)
        
        # Add theme filter if specified
        if themes:
            placeholders = [f":theme_{i}" for i in range(len(themes))]
            theme_params = {f"theme_{i}": theme for i, theme in enumerate(themes)}
            where_clause += f" AND theme IN ({', '.join(placeholders)})"
            params.update(theme_params)
        
        # Add evaluation prompt filter if specified
        if evaluation_prompt_ids:
            placeholders = [f":prompt_{i}" for i in range(len(evaluation_prompt_ids))]
            prompt_params = {f"prompt_{i}": prompt for i, prompt in enumerate(evaluation_prompt_ids)}
            where_clause += f" AND evaluation_prompt_id IN ({', '.join(placeholders)})"
            params.update(prompt_params)
        
        # Add score range filter if specified
        if min_score is not None:
            where_clause += " AND score >= :min_score"
            params["min_score"] = min_score
        if max_score is not None:
            where_clause += " AND score <= :max_score"
            params["max_score"] = max_score
        
        # Complete queries
        count_query += where_clause
        data_query += where_clause
        
        # Add order by and pagination to data query
        data_query += f"""
        ORDER BY 
            {sort_by} {'DESC' if sort_desc else 'ASC'}
        LIMIT :limit OFFSET :offset
        """
        
        # Execute count query
        try:
            count_result = await self.execute_query(count_query, params)
            total_count = count_result[0]['total'] if count_result else 0
            
            # Execute data query
            results = await self.execute_query(data_query, params)
            
            # Convert timestamps to ISO format
            for result in results:
                if 'timestamp' in result and result['timestamp']:
                    result['timestamp'] = result['timestamp'].isoformat()
                
                # Convert result_metadata to dict if it's JSON string
                if 'result_metadata' in result and result['result_metadata']:
                    if isinstance(result['result_metadata'], str):
                        try:
                            result['result_metadata'] = json.loads(result['result_metadata'])
                        except:
                            pass
            
            return results, total_count
        except Exception as e:
            self.logger.error(f"Error getting filtered results: {str(e)}")
            return [], 0
    
    async def get_available_themes(self) -> List[str]:
        """Get list of available themes in the database."""
        query = "SELECT DISTINCT theme FROM evaluation_results ORDER BY theme"
        try:
            results = await self.execute_query(query)
            return [result['theme'] for result in results if result['theme']]
        except Exception as e:
            self.logger.error(f"Error getting available themes: {str(e)}")
            return []
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models in the database."""
        query = "SELECT DISTINCT model_id FROM evaluation_results ORDER BY model_id"
        try:
            results = await self.execute_query(query)
            return [result['model_id'] for result in results if result['model_id']]
        except Exception as e:
            self.logger.error(f"Error getting available models: {str(e)}")
            return []
    
    async def get_available_evaluation_prompts(self) -> List[Dict[str, str]]:
        """Get list of available evaluation prompts in the database."""
        query = """
        SELECT DISTINCT evaluation_prompt_id, evaluation_prompt 
        FROM evaluation_results 
        WHERE evaluation_prompt_id IS NOT NULL
        ORDER BY evaluation_prompt_id
        """
        try:
            results = await self.execute_query(query)
            return [{
                'id': result['evaluation_prompt_id'],
                'prompt': result['evaluation_prompt']
            } for result in results if result['evaluation_prompt_id']]
        except Exception as e:
            self.logger.error(f"Error getting available evaluation prompts: {str(e)}")
            return []
    
    async def get_dashboard_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        models: Optional[List[str]] = None,
        themes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get dashboard summary statistics.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            models: List of model IDs to include
            themes: List of themes to include
            
        Returns:
            Dictionary with summary statistics
        """
        # Default date range if not specified (last 30 days)
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Build query with filters
        query = """
        SELECT 
            COUNT(*) as total_evaluations,
            COUNT(DISTINCT model_id) as total_models,
            COUNT(DISTINCT theme) as total_themes,
            COUNT(DISTINCT evaluation_prompt_id) as total_prompts,
            AVG(score) as avg_score,
            MIN(score) as min_score,
            MAX(score) as max_score
        FROM 
            evaluation_results
        WHERE 
            timestamp BETWEEN :start_date AND :end_date
        """
        
        params = {
            "start_date": start_date,
            "end_date": end_date
        }
        
        # Add model filter if specified
        if models:
            placeholders = [f":model_{i}" for i in range(len(models))]
            model_params = {f"model_{i}": model for i, model in enumerate(models)}
            query += f" AND model_id IN ({', '.join(placeholders)})"
            params.update(model_params)
        
        # Add theme filter if specified
        if themes:
            placeholders = [f":theme_{i}" for i in range(len(themes))]
            theme_params = {f"theme_{i}": theme for i, theme in enumerate(themes)}
            query += f" AND theme IN ({', '.join(placeholders)})"
            params.update(theme_params)
        
        # Execute query
        try:
            results = await self.execute_query(query, params)
            if not results:
                return {}
            
            summary = results[0]
            
            # Get top performing models
            top_models_query = """
            SELECT 
                model_id,
                AVG(score) as avg_score,
                COUNT(*) as eval_count
            FROM 
                evaluation_results
            WHERE 
                timestamp BETWEEN :start_date AND :end_date
            """
            
            # Apply same filters
            if models:
                top_models_query += f" AND model_id IN ({', '.join(placeholders)})"
            if themes:
                top_models_query += f" AND theme IN ({', '.join([f':theme_{i}' for i in range(len(themes))])})"
            
            top_models_query += """
            GROUP BY 
                model_id
            ORDER BY 
                avg_score DESC
            LIMIT 5
            """
            
            top_models = await self.execute_query(top_models_query, params)
            
            # Get top performing themes
            top_themes_query = """
            SELECT 
                theme,
                AVG(score) as avg_score,
                COUNT(*) as eval_count
            FROM 
                evaluation_results
            WHERE 
                timestamp BETWEEN :start_date AND :end_date
            """
            
            # Apply same filters
            if models:
                top_themes_query += f" AND model_id IN ({', '.join(placeholders)})"
            if themes:
                top_themes_query += f" AND theme IN ({', '.join([f':theme_{i}' for i in range(len(themes))])})"
            
            top_themes_query += """
            GROUP BY 
                theme
            ORDER BY 
                avg_score DESC
            LIMIT 5
            """
            
            top_themes = await self.execute_query(top_themes_query, params)
            
            # Get recent trend (last 7 days vs previous 7 days)
            if (end_date - start_date).days >= 14:
                mid_date = end_date - timedelta(days=7)
                
                recent_trend_query = """
                SELECT 
                    CASE 
                        WHEN timestamp BETWEEN :mid_date AND :end_date THEN 'recent'
                        WHEN timestamp BETWEEN :start_date AND :mid_date THEN 'previous'
                    END as period,
                    AVG(score) as avg_score,
                    COUNT(*) as eval_count
                FROM 
                    evaluation_results
                WHERE 
                    timestamp BETWEEN :start_date AND :end_date
                """
                
                # Apply same filters
                if models:
                    recent_trend_query += f" AND model_id IN ({', '.join(placeholders)})"
                if themes:
                    recent_trend_query += f" AND theme IN ({', '.join([f':theme_{i}' for i in range(len(themes))])})"
                
                recent_trend_query += """
                GROUP BY 
                    period
                ORDER BY 
                    period
                """
                
                trend_params = params.copy()
                trend_params["mid_date"] = mid_date
                
                trend_results = await self.execute_query(recent_trend_query, trend_params)
                
                trend_data = {}
                for result in trend_results:
                    trend_data[result['period']] = {
                        'avg_score': result['avg_score'],
                        'eval_count': result['eval_count']
                    }
                
                if 'recent' in trend_data and 'previous' in trend_data:
                    trend_change = trend_data['recent']['avg_score'] - trend_data['previous']['avg_score']
                    summary['trend_change'] = trend_change
                    summary['trend_percentage'] = (trend_change / trend_data['previous']['avg_score'] * 100) if trend_data['previous']['avg_score'] else 0
            
            return {
                "summary": summary,
                "top_models": top_models,
                "top_themes": top_themes
            }
        except Exception as e:
            self.logger.error(f"Error getting dashboard summary: {str(e)}")
            return {}
