"""
Storage Layer for LLM Metrics and Analysis

This module provides:
1. SQLite integration for structured data storage
2. InfluxDB integration for time-series metrics
3. Helper functions to manage persistent data
"""

import os
import json
import uuid
import logging
import sqlite3
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try importing InfluxDB
try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    logger.warning("InfluxDB client not available. Install with 'pip install influxdb-client'")
    INFLUXDB_AVAILABLE = False


# --- SQLite Storage ---

class SQLiteStorage:
    """
    SQLite-based storage for LLM responses, prompts, and batch information.
    Provides persistent, structured storage for non-time-series data.
    """
    
    def __init__(self, db_path: str = "./llm_data.sqlite"):
        """
        Initialize SQLite storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.initialize()
    
    def initialize(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        self.conn = sqlite3.connect(self.db_path)
        
        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Create tables if they don't exist
        self._create_tables()
        
        logger.info(f"SQLite database initialized at {self.db_path}")
    
    def _create_tables(self) -> None:
        """Create the database tables."""
        cursor = self.conn.cursor()
        
        # Models table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            provider TEXT,
            version TEXT,
            first_used TIMESTAMP,
            last_used TIMESTAMP,
            metadata TEXT
        )
        ''')
        
        # Batches table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS batches (
            id TEXT PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            description TEXT,
            metadata TEXT
        )
        ''')
        
        # Prompts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            batch_id TEXT,
            prompt_idx INTEGER,
            category TEXT,
            metadata TEXT,
            FOREIGN KEY (batch_id) REFERENCES batches(id)
        )
        ''')
        
        # Responses table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS responses (
            id TEXT PRIMARY KEY,
            model_id INTEGER NOT NULL,
            prompt_id INTEGER NOT NULL,
            batch_id TEXT NOT NULL,
            text TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            embedding_id TEXT,
            metadata TEXT,
            FOREIGN KEY (model_id) REFERENCES models(id),
            FOREIGN KEY (prompt_id) REFERENCES prompts(id),
            FOREIGN KEY (batch_id) REFERENCES batches(id)
        )
        ''')
        
        # Metrics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            batch_id TEXT NOT NULL,
            model_id INTEGER,
            timestamp TIMESTAMP NOT NULL,
            value REAL NOT NULL,
            sample_count INTEGER NOT NULL,
            success BOOLEAN NOT NULL,
            error TEXT,
            metadata TEXT,
            details TEXT,
            FOREIGN KEY (batch_id) REFERENCES batches(id),
            FOREIGN KEY (model_id) REFERENCES models(id)
        )
        ''')
        
        self.conn.commit()
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def store_batch(self, batch_id: str, timestamp: datetime.datetime,
                   description: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store batch information.
        
        Args:
            batch_id: Unique identifier for the batch
            timestamp: When the batch was created
            description: Optional description of the batch
            metadata: Optional additional metadata as a dictionary
        """
        cursor = self.conn.cursor()
        
        # Convert metadata to JSON if provided
        metadata_json = json.dumps(metadata) if metadata else None
        
        # Insert or update batch
        cursor.execute('''
        INSERT OR REPLACE INTO batches (id, timestamp, description, metadata)
        VALUES (?, ?, ?, ?)
        ''', (batch_id, timestamp.isoformat(), description, metadata_json))
        
        self.conn.commit()
    
    def store_model(self, name: str, provider: Optional[str] = None,
                  version: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Store model information.
        
        Args:
            name: Model name
            provider: Provider name (e.g., OpenAI, Anthropic)
            version: Model version
            metadata: Optional additional metadata
            
        Returns:
            Model ID in the database
        """
        cursor = self.conn.cursor()
        
        # Convert metadata to JSON if provided
        metadata_json = json.dumps(metadata) if metadata else None
        
        now = datetime.datetime.now().isoformat()
        
        # Check if model exists
        cursor.execute("SELECT id FROM models WHERE name = ?", (name,))
        result = cursor.fetchone()
        
        if result:
            # Update existing model
            model_id = result[0]
            cursor.execute('''
            UPDATE models SET last_used = ?, provider = ?, version = ?, metadata = ?
            WHERE id = ?
            ''', (now, provider, version, metadata_json, model_id))
        else:
            # Insert new model
            cursor.execute('''
            INSERT INTO models (name, provider, version, first_used, last_used, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (name, provider, version, now, now, metadata_json))
            model_id = cursor.lastrowid
        
        self.conn.commit()
        return model_id
    
    def store_prompt(self, text: str, batch_id: str, prompt_idx: int,
                    category: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Store prompt information.
        
        Args:
            text: Prompt text
            batch_id: ID of the batch this prompt belongs to
            prompt_idx: Index of this prompt in the batch
            category: Optional category or theme of the prompt
            metadata: Optional additional metadata
            
        Returns:
            Prompt ID in the database
        """
        cursor = self.conn.cursor()
        
        # Convert metadata to JSON if provided
        metadata_json = json.dumps(metadata) if metadata else None
        
        # Check if prompt exists for this batch and index
        cursor.execute(
            "SELECT id FROM prompts WHERE batch_id = ? AND prompt_idx = ?",
            (batch_id, prompt_idx)
        )
        result = cursor.fetchone()
        
        if result:
            # Update existing prompt
            prompt_id = result[0]
            cursor.execute('''
            UPDATE prompts SET text = ?, category = ?, metadata = ?
            WHERE id = ?
            ''', (text, category, metadata_json, prompt_id))
        else:
            # Insert new prompt
            cursor.execute('''
            INSERT INTO prompts (text, batch_id, prompt_idx, category, metadata)
            VALUES (?, ?, ?, ?, ?)
            ''', (text, batch_id, prompt_idx, category, metadata_json))
            prompt_id = cursor.lastrowid
        
        self.conn.commit()
        return prompt_id
    
    def store_response(self, response_id: str, model_name: str, prompt_text: str,
                      batch_id: str, prompt_idx: int, response_text: str,
                      timestamp: datetime.datetime, embedding_id: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store an LLM response.
        
        Args:
            response_id: Unique identifier for the response
            model_name: Name of the model that generated the response
            prompt_text: Text of the prompt
            batch_id: ID of the batch this response belongs to
            prompt_idx: Index of the prompt in the batch
            response_text: The response text from the LLM
            timestamp: When the response was generated
            embedding_id: Optional ID of the embedding in ChromaDB
            metadata: Optional additional metadata
        """
        cursor = self.conn.cursor()
        
        # Make sure model exists
        model_id = self.store_model(model_name)
        
        # Make sure prompt exists
        prompt_id = self.store_prompt(prompt_text, batch_id, prompt_idx)
        
        # Convert metadata to JSON if provided
        metadata_json = json.dumps(metadata) if metadata else None
        
        # Insert or update response
        cursor.execute('''
        INSERT OR REPLACE INTO responses
        (id, model_id, prompt_id, batch_id, text, timestamp, embedding_id, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            response_id, model_id, prompt_id, batch_id,
            response_text, timestamp.isoformat(), embedding_id, metadata_json
        ))
        
        self.conn.commit()
    
    def store_metric_result(self, metric_id: str, name: str, batch_id: str,
                          model_name: Optional[str], timestamp: datetime.datetime,
                          value: float, sample_count: int, success: bool = True,
                          error: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                          details: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a metric result.
        
        Args:
            metric_id: Unique identifier for the metric result
            name: Name of the metric
            batch_id: ID of the batch this metric was calculated for
            model_name: Optional name of the model (None for batch-level metrics)
            timestamp: When the metric was calculated
            value: Primary numeric result value
            sample_count: Number of samples used in the calculation
            success: Whether the metric calculation succeeded
            error: Optional error message if calculation failed
            metadata: Optional additional metadata
            details: Optional detailed results or breakdown
        """
        cursor = self.conn.cursor()
        
        # Get model ID if model name is provided
        model_id = None
        if model_name:
            model_id = self.store_model(model_name)
        
        # Convert metadata and details to JSON if provided
        metadata_json = json.dumps(metadata) if metadata else None
        details_json = json.dumps(details) if details else None
        
        # Insert or update metric result
        cursor.execute('''
        INSERT OR REPLACE INTO metrics
        (id, name, batch_id, model_id, timestamp, value, sample_count, success, error, metadata, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metric_id, name, batch_id, model_id, timestamp.isoformat(),
            value, sample_count, success, error, metadata_json, details_json
        ))
        
        self.conn.commit()
    
    def get_batches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get information about recent batches.
        
        Args:
            limit: Maximum number of batches to return
            
        Returns:
            List of batch information dictionaries
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
        SELECT id, timestamp, description, metadata
        FROM batches
        ORDER BY timestamp DESC
        LIMIT ?
        ''', (limit,))
        
        batches = []
        for row in cursor.fetchall():
            batch_id, timestamp_str, description, metadata_json = row
            
            # Parse metadata JSON
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            batches.append({
                "id": batch_id,
                "timestamp": timestamp_str,
                "description": description,
                "metadata": metadata
            })
        
        return batches
    
    def get_batch_details(self, batch_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific batch.
        
        Args:
            batch_id: ID of the batch
            
        Returns:
            Dictionary with batch details
        """
        cursor = self.conn.cursor()
        
        # Get batch info
        cursor.execute('''
        SELECT id, timestamp, description, metadata
        FROM batches
        WHERE id = ?
        ''', (batch_id,))
        
        batch_row = cursor.fetchone()
        if not batch_row:
            return {}
        
        batch_id, timestamp_str, description, metadata_json = batch_row
        
        # Get prompts
        cursor.execute('''
        SELECT id, text, prompt_idx, category
        FROM prompts
        WHERE batch_id = ?
        ORDER BY prompt_idx
        ''', (batch_id,))
        
        prompts = []
        for row in cursor.fetchall():
            prompt_id, text, prompt_idx, category = row
            prompts.append({
                "id": prompt_id,
                "text": text,
                "index": prompt_idx,
                "category": category
            })
        
        # Get models used
        cursor.execute('''
        SELECT DISTINCT m.id, m.name, m.provider, m.version
        FROM responses r
        JOIN models m ON r.model_id = m.id
        WHERE r.batch_id = ?
        ''', (batch_id,))
        
        models = []
        for row in cursor.fetchall():
            model_id, name, provider, version = row
            models.append({
                "id": model_id,
                "name": name,
                "provider": provider,
                "version": version
            })
        
        # Get response count
        cursor.execute('''
        SELECT COUNT(*) FROM responses WHERE batch_id = ?
        ''', (batch_id,))
        
        response_count = cursor.fetchone()[0]
        
        # Get metric summary
        cursor.execute('''
        SELECT name, COUNT(*), AVG(value)
        FROM metrics
        WHERE batch_id = ?
        GROUP BY name
        ''', (batch_id,))
        
        metrics = []
        for row in cursor.fetchall():
            name, count, avg_value = row
            metrics.append({
                "name": name,
                "count": count,
                "average_value": avg_value
            })
        
        # Compile and return details
        return {
            "id": batch_id,
            "timestamp": timestamp_str,
            "description": description,
            "metadata": json.loads(metadata_json) if metadata_json else {},
            "prompt_count": len(prompts),
            "prompts": prompts,
            "model_count": len(models),
            "models": models,
            "response_count": response_count,
            "metrics": metrics
        }
    
    def get_metrics_for_batch(self, batch_id: str, metric_name: Optional[str] = None,
                            model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get metric results for a specific batch.
        
        Args:
            batch_id: ID of the batch
            metric_name: Optional filter by metric name
            model_name: Optional filter by model name
            
        Returns:
            List of metric result dictionaries
        """
        cursor = self.conn.cursor()
        
        query = '''
        SELECT m.id, m.name, m.batch_id, mo.name as model_name, 
               m.timestamp, m.value, m.sample_count, m.success, 
               m.error, m.metadata, m.details
        FROM metrics m
        LEFT JOIN models mo ON m.model_id = mo.id
        WHERE m.batch_id = ?
        '''
        
        params = [batch_id]
        
        if metric_name:
            query += " AND m.name = ?"
            params.append(metric_name)
        
        if model_name:
            query += " AND mo.name = ?"
            params.append(model_name)
        
        query += " ORDER BY m.timestamp DESC"
        
        cursor.execute(query, params)
        
        metrics = []
        for row in cursor.fetchall():
            (metric_id, name, batch_id, model_name, timestamp_str, value, 
             sample_count, success, error, metadata_json, details_json) = row
            
            # Parse JSON fields
            metadata = json.loads(metadata_json) if metadata_json else {}
            details = json.loads(details_json) if details_json else {}
            
            metrics.append({
                "id": metric_id,
                "name": name,
                "batch_id": batch_id,
                "model_name": model_name,
                "timestamp": timestamp_str,
                "value": value,
                "sample_count": sample_count,
                "success": bool(success),
                "error": error,
                "metadata": metadata,
                "details": details
            })
        
        return metrics
    
    def get_responses_by_prompt(self, batch_id: str, prompt_idx: int) -> List[Dict[str, Any]]:
        """
        Get all responses for a specific prompt in a batch.
        
        Args:
            batch_id: ID of the batch
            prompt_idx: Index of the prompt
            
        Returns:
            List of response dictionaries
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
        SELECT r.id, m.name as model_name, p.text as prompt_text, r.text as response_text,
               r.timestamp, r.embedding_id, r.metadata
        FROM responses r
        JOIN prompts p ON r.prompt_id = p.id
        JOIN models m ON r.model_id = m.id
        WHERE r.batch_id = ? AND p.prompt_idx = ?
        ORDER BY r.timestamp
        ''', (batch_id, prompt_idx))
        
        responses = []
        for row in cursor.fetchall():
            (response_id, model_name, prompt_text, response_text,
             timestamp_str, embedding_id, metadata_json) = row
            
            # Parse metadata JSON
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            responses.append({
                "id": response_id,
                "model_name": model_name,
                "prompt": prompt_text,
                "response": response_text,
                "timestamp": timestamp_str,
                "embedding_id": embedding_id,
                "metadata": metadata
            })
        
        return responses


# --- InfluxDB Storage ---

class InfluxDBStorage:
    """
    InfluxDB-based storage for time-series metrics data.
    """
    
    def __init__(self, url: str = "http://localhost:8086", token: Optional[str] = None,
                org: str = "llm_metrics", bucket: str = "llm_metrics"):
        """
        Initialize InfluxDB storage.
        
        Args:
            url: InfluxDB server URL
            token: Authentication token
            org: Organization name
            bucket: Bucket name
        """
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.client = None
        self.write_api = None
        self.initialized = False
        
        # Initialize connection
        self.initialize()
    
    def initialize(self) -> None:
        """Initialize the InfluxDB client and write API."""
        if not INFLUXDB_AVAILABLE:
            logger.warning("InfluxDB client not available. Skipping initialization.")
            return
        
        try:
            self.client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org
            )
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.initialized = True
            logger.info(f"Connected to InfluxDB at {self.url}")
        except Exception as e:
            logger.error(f"Failed to initialize InfluxDB client: {str(e)}")
            self.initialized = False
    
    def close(self) -> None:
        """Close the InfluxDB client and cleanup resources."""
        if self.write_api:
            try:
                self.write_api.close()
            except Exception as e:
                logger.error(f"Error closing write API: {str(e)}")
            self.write_api = None
            
        if self.client:
            try:
                self.client.close()
            except Exception as e:
                logger.error(f"Error closing InfluxDB client: {str(e)}")
            self.client = None
            
        self.initialized = False
    
    def store_metric(self, metric_name: str, value: float, batch_id: str, 
                    model_name: Optional[str] = None, timestamp: Optional[datetime.datetime] = None,
                    tags: Optional[Dict[str, str]] = None,
                    fields: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a metric data point in InfluxDB.
        
        Args:
            metric_name: Name of the metric
            value: Primary metric value
            batch_id: ID of the batch
            model_name: Optional name of the model
            timestamp: Optional timestamp (uses current time if not provided)
            tags: Optional additional tags for the data point
            fields: Optional additional fields for the data point
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not INFLUXDB_AVAILABLE or not self.client:
            return False
        
        try:
            # Create a data point
            point = Point("llm_metric")
            
            # Add tags
            point = point.tag("metric", metric_name)
            point = point.tag("batch_id", batch_id)
            
            if model_name:
                point = point.tag("model", model_name)
            
            # Add custom tags
            if tags:
                for key, value in tags.items():
                    point = point.tag(key, value)
            
            # Add fields
            point = point.field("value", value)
            
            # Add custom fields
            if fields:
                for key, value in fields.items():
                    if isinstance(value, (int, float, bool, str)):
                        point = point.field(key, value)
            
            # Set timestamp
            if timestamp:
                point = point.time(timestamp, WritePrecision.NS)
            
            # Write the point
            self.write_api.write(bucket=self.bucket, record=point)
            return True
            
        except Exception as e:
            logger.error(f"Error storing metric in InfluxDB: {str(e)}")
            return False
    
    def store_batch_metrics(self, metrics: List[Dict[str, Any]]) -> int:
        """
        Store multiple metric results at once.
        
        Args:
            metrics: List of metric dictionaries
            
        Returns:
            Number of metrics successfully stored
        """
        if not INFLUXDB_AVAILABLE or not self.client:
            return 0
        
        try:
            points = []
            for metric in metrics:
                # Extract required fields
                metric_name = metric.get("name")
                value = metric.get("value")
                batch_id = metric.get("batch_id")
                
                if not all([metric_name, value is not None, batch_id]):
                    logger.warning(f"Skipping metric with missing required fields: {metric}")
                    continue
                
                # Create a data point
                point = Point("llm_metric")
                
                # Add tags
                point = point.tag("metric", metric_name)
                point = point.tag("batch_id", batch_id)
                
                # Add optional tags
                model_name = metric.get("model_name")
                if model_name:
                    point = point.tag("model", model_name)
                
                # Get additional tags from metadata
                metadata = metric.get("metadata", {})
                for key, tag_value in metadata.items():
                    if isinstance(tag_value, str):
                        point = point.tag(key, tag_value)
                
                # Add fields
                point = point.field("value", float(value))
                point = point.field("sample_count", int(metric.get("sample_count", 0)))
                point = point.field("success", bool(metric.get("success", True)))
                
                # Set timestamp
                timestamp = metric.get("timestamp")
                if timestamp:
                    if isinstance(timestamp, str):
                        timestamp = datetime.datetime.fromisoformat(timestamp)
                    point = point.time(timestamp, WritePrecision.NS)
                
                points.append(point)
            
            # Write all points
            if points:
                self.write_api.write(bucket=self.bucket, record=points)
            
            return len(points)
            
        except Exception as e:
            logger.error(f"Error storing batch metrics in InfluxDB: {str(e)}")
            return 0


# --- Storage Manager ---

class StorageManager:
    """
    Unified manager for all storage backends.
    """
    
    def __init__(self, sqlite_path: str = "./llm_data.sqlite",
               influxdb_url: str = "http://localhost:8086",
               influxdb_token: Optional[str] = None,
               influxdb_org: str = "llm_metrics",
               influxdb_bucket: str = "llm_metrics"):
        """
        Initialize the storage manager.
        
        Args:
            sqlite_path: Path to SQLite database file
            influxdb_url: InfluxDB server URL
            influxdb_token: InfluxDB authentication token
            influxdb_org: InfluxDB organization name
            influxdb_bucket: InfluxDB bucket name
        """
        # Create storage instances
        self.sqlite = SQLiteStorage(db_path=sqlite_path)
        
        self.influxdb = None
        if INFLUXDB_AVAILABLE:
            self.influxdb = InfluxDBStorage(
                url=influxdb_url,
                token=influxdb_token,
                org=influxdb_org,
                bucket=influxdb_bucket
            )
    
    def close(self) -> None:
        """Close all storage connections."""
        self.sqlite.close()
        if self.influxdb:
            self.influxdb.close()
    
    def store_batch_from_chroma(self, collection, batch_id: str,
                               description: Optional[str] = None) -> int:
        """
        Store batch data from ChromaDB into structured storage.
        
        Args:
            collection: ChromaDB collection
            batch_id: ID of the batch to store
            description: Optional description of the batch
            
        Returns:
            Number of responses stored
        """
        # Helper function to get batch responses from ChromaDB
        def get_batch_responses(collection, batch_id):
            results = collection.get(where={"batch_id": batch_id})
            
            formatted_results = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents']):
                    result_entry = {
                        "text": doc,
                        "metadata": results['metadatas'][i] if results['metadatas'] else {},
                        "id": results['ids'][i] if results['ids'] else None,
                    }
                    formatted_results.append(result_entry)
            
            return formatted_results
        
        # Get responses from ChromaDB
        responses = get_batch_responses(collection, batch_id)
        
        if not responses:
            logger.warning(f"No responses found for batch {batch_id}")
            return 0
        
        # Get timestamp from first response
        first_resp = responses[0]
        timestamp_str = first_resp['metadata'].get('timestamp')
        
        if timestamp_str:
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp_str)
            except ValueError:
                timestamp = datetime.datetime.now()
        else:
            timestamp = datetime.datetime.now()
        
        # Store batch info
        self.sqlite.store_batch(
            batch_id=batch_id,
            timestamp=timestamp,
            description=description,
            metadata={"source": "chromadb", "response_count": len(responses)}
        )
        
        # Track prompts to avoid duplicates
        processed_prompts = set()
        
        # Store each response
        for response in responses:
            metadata = response['metadata']
            model_name = metadata.get('model_name', 'unknown')
            prompt = metadata.get('prompt', '')
            prompt_idx = metadata.get('prompt_idx', 0)
            response_text = response['text']
            response_id = response['id']
            
            # Store the prompt if we haven't seen it before
            prompt_key = f"{batch_id}:{prompt_idx}"
            if prompt_key not in processed_prompts:
                self.sqlite.store_prompt(
                    text=prompt,
                    batch_id=batch_id,
                    prompt_idx=prompt_idx
                )
                processed_prompts.add(prompt_key)
            
            # Store the response
            self.sqlite.store_response(
                response_id=response_id,
                model_name=model_name,
                prompt_text=prompt,
                batch_id=batch_id,
                prompt_idx=prompt_idx,
                response_text=response_text,
                timestamp=timestamp,
                embedding_id=response_id,
                metadata=metadata
            )
        
        return len(responses)
    
    def store_metrics(self, metrics: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Store metric results in both SQLite and InfluxDB.
        
        Args:
            metrics: List of metric result dictionaries
            
        Returns:
            Tuple of (sqlite_count, influxdb_count)
        """
        sqlite_count = 0
        influxdb_count = 0
        
        # Store in SQLite
        for metric in metrics:
            # Validate required fields
            required_fields = ["metric_name", "batch_id", "value", "sample_count"]
            missing_fields = [field for field in required_fields if field not in metric or metric[field] is None]
            
            if missing_fields:
                logger.warning(f"Skipping metric due to missing required fields: {missing_fields}")
                continue
                
            metric_id = metric.get("result_id", str(uuid.uuid4()))
            name = metric["metric_name"]
            batch_id = metric["batch_id"]
            model_name = metric.get("model_name")
            timestamp = metric.get("timestamp", datetime.datetime.now())
            value = metric["value"]
            sample_count = metric["sample_count"]
            success = metric.get("success", True)
            error = metric.get("error")
            metadata = metric.get("metadata", {})
            details = metric.get("details", {})
            
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.datetime.now()
            
            try:
                self.sqlite.store_metric_result(
                    metric_id=metric_id,
                    name=name,
                    batch_id=batch_id,
                    model_name=model_name,
                    timestamp=timestamp,
                    value=value,
                    sample_count=sample_count,
                    success=success,
                    error=error,
                    metadata=metadata,
                    details=details
                )
                sqlite_count += 1
            except Exception as e:
                logger.error(f"Failed to store metric in SQLite: {str(e)}")
                continue
        
        # Store in InfluxDB if available
        if self.influxdb:
            try:
                influxdb_count = self.influxdb.store_batch_metrics(metrics)
            except Exception as e:
                logger.error(f"Failed to store metrics in InfluxDB: {str(e)}")
        
        return sqlite_count, influxdb_count


# --- Helper functions ---

def setup_storage(base_dir: str = "./llm_data") -> StorageManager:
    """
    Set up and initialize storage.
    
    Args:
        base_dir: Base directory for all data storage
        
    Returns:
        Storage manager instance
    """
    # Make sure directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Set up paths
    sqlite_path = os.path.join(base_dir, "llm_data.sqlite")
    
    # Create and return storage manager
    return StorageManager(
        sqlite_path=sqlite_path,
        influxdb_url="http://localhost:8086",  # Default local InfluxDB URL
        influxdb_token=None,  # No auth for local dev instances
        influxdb_org="llm_metrics",
        influxdb_bucket="llm_metrics"
    )


def import_from_chromadb(storage_manager: StorageManager, collection, 
                       max_batches: int = None) -> int:
    """
    Import all batches from ChromaDB into structured storage.
    
    Args:
        storage_manager: Storage manager instance
        collection: ChromaDB collection
        max_batches: Maximum number of batches to import
        
    Returns:
        Number of batches imported
    """
    # Helper function to get batch IDs from ChromaDB
    def get_batch_ids(collection):
        results = collection.get()
        
        if not results or not results['metadatas']:
            return []
        
        # Extract unique batch IDs
        batch_ids = set()
        for metadata in results['metadatas']:
            batch_id = metadata.get('batch_id')
            if batch_id:
                batch_ids.add(batch_id)
        
        return list(batch_ids)
    
    # Get all batch IDs
    batch_ids = get_batch_ids(collection)
    
    if max_batches:
        batch_ids = batch_ids[:max_batches]
    
    imported_count = 0
    for batch_id in batch_ids:
        responses_stored = storage_manager.store_batch_from_chroma(
            collection=collection,
            batch_id=batch_id,
            description=f"Imported from ChromaDB on {datetime.datetime.now().isoformat()}"
        )
        
        if responses_stored > 0:
            imported_count += 1
            logger.info(f"Imported batch {batch_id} with {responses_stored} responses")
    
    return imported_count


def setup_influxdb_locally(data_dir: str = "./influxdb_data") -> None:
    """
    Generate a docker-compose file for setting up InfluxDB locally.
    
    Args:
        data_dir: Directory for InfluxDB data
    """
    compose_file = "docker-compose-influxdb.yml"
    
    # Docker compose content
    content = f"""version: '3'

services:
  influxdb:
    image: influxdb:2.7
    ports:
      - "8086:8086"
    volumes:
      - {data_dir}:/var/lib/influxdb2
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME=admin
      - DOCKER_INFLUXDB_INIT_PASSWORD=adminpassword
      - DOCKER_INFLUXDB_INIT_ORG=llm_metrics
      - DOCKER_INFLUXDB_INIT_BUCKET=llm_metrics
      - DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=mytoken
    restart: unless-stopped

  # Uncomment below if you want to include Grafana as well
  # grafana:
  #   image: grafana/grafana:latest
  #   ports:
  #     - "3000:3000"
  #   volumes:
  #     - ./grafana_data:/var/lib/grafana
  #   environment:
  #     - GF_SECURITY_ADMIN_USER=admin
  #     - GF_SECURITY_ADMIN_PASSWORD=admin
  #   restart: unless-stopped
  #   depends_on:
  #     - influxdb
"""
    
    # Write to file
    with open(compose_file, 'w') as f:
        f.write(content)
    
    logger.info(f"Created Docker Compose file: {compose_file}")
    logger.info("Start InfluxDB with: docker-compose -f docker-compose-influxdb.yml up -d")
    logger.info("InfluxDB will be available at: http://localhost:8086")
    logger.info("Username: admin, Password: adminpassword, Token: mytoken")