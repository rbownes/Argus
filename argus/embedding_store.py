"""
Module for embedding and storing LLM responses in a vector database.

This module provides functionality to embed text using Hugging Face models
and store the embeddings in a ChromaDB vector database with rich metadata
for retrieval by model, time, batch ID, and semantic similarity.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import os
import logging
from datetime import datetime, timedelta
import uuid
import json
from pydantic import BaseModel, Field, validator, root_validator, ConfigDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error("sentence_transformers package not found. Please install it with 'pip install sentence-transformers'")
    raise

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    logger.error("chromadb package not found. Please install it with 'pip install chromadb'")
    raise

# Import from local module if available
try:
    from llm_query import LLMResponse
    local_module_available = True
except ImportError:
    logger.warning("Local llm_query module not found. Using simplified data structures.")
    local_module_available = False
    # Define a simplified version for compatibility
    class LLMResponse(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        model: str
        prompt: str
        response_text: str
        timestamp: datetime = Field(default_factory=datetime.now)
        metadata: Dict[str, Any] = Field(default_factory=dict)
        error: Optional[str] = None


class EmbeddingParams(BaseModel):
    """Pydantic model for validating embedding and storage parameters."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model_outputs: Dict[str, List[Any]] = Field(
        ..., description="Dictionary of model names to lists of responses (text or LLMResponse objects)"
    )
    prompts: Optional[List[str]] = Field(
        default=None, description="List of prompts that were used to generate the responses (required if using text responses)"
    )
    embedding_model_name: str = Field(
        default="BAAI/bge-base-en-v1.5", description="HuggingFace model to use for embeddings"
    )
    persist_directory: Optional[str] = Field(
        default="./chroma_db", description="Directory to persist the vector database"
    )
    collection_name: str = Field(
        default="llm_responses", description="Name for the ChromaDB collection"
    )
    batch_id: Optional[str] = Field(
        default=None, description="Unique identifier for this batch of embeddings"
    )
    timestamp: Optional[datetime] = Field(
        default=None, description="Timestamp when these queries were run"
    )
    
    @root_validator(pre=True)
    def validate_model_outputs(cls, values):
        model_outputs = values.get('model_outputs')
        if not model_outputs:
            raise ValueError("Model outputs dictionary cannot be empty")
        
        # Check data type of the values
        for model, responses in model_outputs.items():
            if not responses:
                raise ValueError(f"Responses list for model {model} cannot be empty")
                
            # Check if responses are strings or LLMResponse objects
            first_response = responses[0]
            if isinstance(first_response, str):
                response_type = str
            elif hasattr(first_response, 'model') and hasattr(first_response, 'prompt') and hasattr(first_response, 'response_text'):
                # Check for LLMResponse-like object by checking for required attributes
                response_type = type(first_response)
            else:
                raise ValueError(f"Responses must be either strings or LLMResponse-like objects with model, prompt, and response_text attributes")
                
            # Check that all responses in this list are of the same type
            if not all(isinstance(r, response_type) for r in responses):
                raise ValueError(f"All responses for model {model} must be of the same type")
                
        return values
    
    @validator('prompts')
    def validate_prompts(cls, v, values):
        model_outputs = values.get('model_outputs')
        
        # If model_outputs is a dict of str -> List[str], prompts are required
        if model_outputs:
            first_model = next(iter(model_outputs))
            if model_outputs[first_model] and isinstance(model_outputs[first_model][0], str):
                if not v:
                    raise ValueError("Prompts list is required when response texts are provided as strings")
                if any(len(responses) != len(v) for responses in model_outputs.values()):
                    raise ValueError("Number of responses for each model must match number of prompts")
        
        return v
    
    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.now()
    
    @validator('batch_id', pre=True, always=True)
    def set_batch_id(cls, v):
        return v or f"batch-{uuid.uuid4()}"


def embed_and_store_model_outputs(
    model_outputs: Union[Dict[str, List[str]], Dict[str, List[LLMResponse]]],
    prompts: Optional[List[str]] = None,
    embedding_model_name: str = "BAAI/bge-base-en-v1.5",
    persist_directory: Optional[str] = "./chroma_db",
    collection_name: str = "llm_responses",
    batch_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Tuple[chromadb.Collection, SentenceTransformer, str, datetime]:
    """
    Embeds model outputs using a Hugging Face embedding model and stores them in a ChromaDB vector database.
    
    This function accepts either:
    1. A dictionary mapping model names to lists of text responses, along with a separate list of prompts
    2. A dictionary mapping model names to lists of LLMResponse objects, which already contain prompt info
    
    Args:
        model_outputs: Dictionary mapping model names to lists of responses (text strings or LLMResponse objects)
        prompts: List of prompts that were used to generate the responses (required if using text responses)
        embedding_model_name: Name of the Hugging Face model to use for embeddings
        persist_directory: Directory to persist the ChromaDB database
        collection_name: Name for the ChromaDB collection
        batch_id: Optional unique identifier for this batch of embeddings
        timestamp: Optional timestamp for when these queries were run
        additional_metadata: Optional additional metadata to store with all embeddings
        
    Returns:
        A tuple containing (ChromaDB collection, embedding model, batch_id, timestamp)
        
    Raises:
        ValueError: If parameter validation fails
        RuntimeError: If there's an error during embedding or database operations
    """
    # Validate inputs with Pydantic
    try:
        params = EmbeddingParams(
            model_outputs=model_outputs,
            prompts=prompts,
            embedding_model_name=embedding_model_name,
            persist_directory=persist_directory,
            collection_name=collection_name,
            batch_id=batch_id,
            timestamp=timestamp
        )
    except Exception as e:
        logger.error(f"Parameter validation failed: {e}")
        raise ValueError(f"Parameter validation failed: {e}")
    
    # Ensure we have required parameters
    assert params.embedding_model_name, "Embedding model name must be provided"
    assert params.collection_name, "Collection name must be provided"
    
    try:
        logger.info(f"Starting embedding process with model {params.embedding_model_name}")
        logger.info(f"Batch ID: {params.batch_id}")
        
        # Load the embedding model from Hugging Face
        embedding_model = SentenceTransformer(params.embedding_model_name)
        logger.info(f"Loaded embedding model: {params.embedding_model_name}")
        
        # Format timestamp for storage
        timestamp_str = params.timestamp.isoformat()
        timestamp_unix = int(params.timestamp.timestamp())
        
        # Create the directory for ChromaDB if it doesn't exist
        if params.persist_directory:
            os.makedirs(params.persist_directory, exist_ok=True)
        
        # Initialize the ChromaDB client
        client_type = "PersistentClient" if params.persist_directory else "EphemeralClient"
        logger.info(f"Initializing ChromaDB {client_type}")
        
        chroma_client = chromadb.PersistentClient(
            path=params.persist_directory
        ) if params.persist_directory else chromadb.EphemeralClient()
        
        # Create a new collection or get existing one
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=params.embedding_model_name
        )
        
        collection = chroma_client.get_or_create_collection(
            name=params.collection_name,
            embedding_function=embedding_function,
            metadata={"description": f"LLM responses embedded with {params.embedding_model_name}"}
        )
        logger.info(f"Using collection: {params.collection_name}")
        
        # Determine if we're working with LLMResponse objects or raw text
        first_model = next(iter(params.model_outputs))
        first_response = params.model_outputs[first_model][0]
        using_llm_response_objects = not isinstance(first_response, str)
        
        # Process each model and its responses
        total_embeddings = 0
        
        for model_name, responses in params.model_outputs.items():
            logger.info(f"Processing {len(responses)} responses from model: {model_name}")
            
            ids = []
            documents = []
            metadatas = []
            
            # Process each response for the current model
            for idx, response in enumerate(responses):
                # Generate a unique ID for each response
                response_id = str(uuid.uuid4())
                ids.append(response_id)
                
                # Extract text and metadata based on response type
                if using_llm_response_objects:
                    # Using LLMResponse objects
                    response_text = response.response_text
                    prompt = response.prompt
                    response_timestamp = response.timestamp
                    response_metadata = response.metadata.copy() if response.metadata else {}
                    
                    # Check for error
                    if response.error:
                        logger.warning(f"Response has error: {response.error}")
                        response_metadata["error"] = response.error
                else:
                    # Using raw text responses and separate prompts list
                    response_text = response
                    prompt = params.prompts[idx] if params.prompts and idx < len(params.prompts) else "Unknown prompt"
                    response_timestamp = params.timestamp
                    response_metadata = {}
                
                # Add the response text to documents
                documents.append(response_text)
                
                # Build metadata for this entry, focusing on model and timestamp
                metadata = {
                    "model_name": model_name,
                    "prompt": prompt,
                    "prompt_idx": idx,
                    "batch_id": params.batch_id,
                    "timestamp": timestamp_str,
                    "timestamp_unix": timestamp_unix,
                    "year": params.timestamp.year,
                    "month": params.timestamp.month,
                    "day": params.timestamp.day,
                    "hour": params.timestamp.hour,
                    "minute": params.timestamp.minute
                }
                
                # Add response-specific metadata
                metadata.update(response_metadata)
                
                # Add additional global metadata if provided
                if additional_metadata:
                    metadata.update(additional_metadata)
                
                # Convert any complex data types to strings for ChromaDB compatibility
                for key, value in list(metadata.items()):
                    if not isinstance(value, (str, int, float, bool)):
                        if isinstance(value, datetime):
                            metadata[key] = value.isoformat()
                        else:
                            try:
                                metadata[key] = json.dumps(value)
                            except:
                                metadata[key] = str(value)
                
                metadatas.append(metadata)
            
            # Add the documents, embeddings, and metadata to the collection
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            total_embeddings += len(documents)
            logger.info(f"Added {len(documents)} embeddings for model: {model_name}")
        
        logger.info(f"Embedding process complete. Total embeddings added: {total_embeddings}")
        return collection, embedding_model, params.batch_id, params.timestamp
        
    except Exception as e:
        logger.error(f"Error during embedding or storage: {str(e)}", exc_info=True)
        raise RuntimeError(f"Error during embedding or storage: {str(e)}")


def query_vector_database(
    collection: chromadb.Collection,
    query_text: str,
    n_results: int = 5,
    filter_metadata: Optional[Dict[str, Any]] = None,
    embedding_model: Optional[SentenceTransformer] = None
) -> List[Dict[str, Any]]:
    """
    Query the vector database to retrieve similar responses with metadata filtering.
    
    Args:
        collection: ChromaDB collection containing the embedded responses.
        query_text: Text to find similar responses for.
        n_results: Number of results to return.
        filter_metadata: Optional metadata filter criteria to apply when querying.
        embedding_model: Optional embedding model to use for query.
        
    Returns:
        List of dictionaries containing the retrieved responses and metadata.
    """
    assert collection is not None, "Collection must be provided"
    assert query_text and isinstance(query_text, str), "Query text must be a non-empty string"
    assert n_results > 0, "Number of results must be positive"
    
    try:
        logger.debug(f"Querying collection with text: {query_text[:50]}...")
        
        # Query the collection with optional metadata filtering
        query_params = {
            "query_texts": [query_text],
            "n_results": n_results
        }
        
        if filter_metadata:
            query_params["where"] = filter_metadata
            logger.debug(f"Using metadata filter: {filter_metadata}")
        
        results = collection.query(**query_params)
        
        # Format the results for easy consumption
        formatted_results = []
        if results and results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                result_entry = {
                    "text": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                    "id": results['ids'][0][i] if results['ids'] and results['ids'][0] else None,
                    "distance": results['distances'][0][i] if results['distances'] and results['distances'][0] else None
                }
                formatted_results.append(result_entry)
        
        logger.debug(f"Retrieved {len(formatted_results)} results")
        return formatted_results
    
    except Exception as e:
        logger.error(f"Error querying vector database: {str(e)}")
        raise RuntimeError(f"Error querying vector database: {str(e)}")


def get_responses_by_model(
    collection: chromadb.Collection,
    model_name: str,
    query_text: Optional[str] = None,
    n_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Retrieve responses from a specific model.
    
    Args:
        collection: ChromaDB collection containing the embedded responses.
        model_name: Name of the model to filter by.
        query_text: Optional text to search for. If None, returns most recent responses.
        n_results: Number of results to return.
        
    Returns:
        List of dictionaries containing the retrieved responses and metadata.
    """
    filter_metadata = {"model_name": model_name}
    
    if query_text:
        return query_vector_database(
            collection=collection,
            query_text=query_text,
            n_results=n_results,
            filter_metadata=filter_metadata
        )
    else:
        # If no query text, return most recent responses for this model
        results = collection.get(
            where=filter_metadata,
            limit=n_results
        )
        
        formatted_results = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents']):
                result_entry = {
                    "text": doc,
                    "metadata": results['metadatas'][i] if results['metadatas'] else {},
                    "id": results['ids'][i] if results['ids'] else None,
                }
                formatted_results.append(result_entry)
                
        # Sort by timestamp (newest first)
        formatted_results.sort(key=lambda x: x['metadata'].get('timestamp_unix', 0), reverse=True)
        return formatted_results


def get_responses_by_time_range(
    collection: chromadb.Collection,
    start_time: datetime,
    end_time: Optional[datetime] = None,
    model_name: Optional[str] = None,
    query_text: Optional[str] = None,
    n_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Retrieve responses from a specific time range.
    
    Args:
        collection: ChromaDB collection containing the embedded responses.
        start_time: Start of the time range to filter by.
        end_time: End of the time range. If None, uses current time.
        model_name: Optional model name to filter by.
        query_text: Optional text to search for. If None, returns based on time only.
        n_results: Number of results to return.
        
    Returns:
        List of dictionaries containing the retrieved responses and metadata.
    """
    if end_time is None:
        end_time = datetime.now()
    
    start_unix = int(start_time.timestamp())
    end_unix = int(end_time.timestamp())
    
    # Build filter
    time_filter = {
        "$and": [
            {"timestamp_unix": {"$gte": start_unix}},
            {"timestamp_unix": {"$lte": end_unix}}
        ]
    }
    
    # Add model filter if specified
    if model_name:
        time_filter["$and"].append({"model_name": model_name})
    
    if query_text:
        return query_vector_database(
            collection=collection,
            query_text=query_text,
            n_results=n_results,
            filter_metadata=time_filter
        )
    else:
        # If no query text, return results based on time range only
        results = collection.get(
            where=time_filter,
            limit=n_results
        )
        
        formatted_results = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents']):
                result_entry = {
                    "text": doc,
                    "metadata": results['metadatas'][i] if results['metadatas'] else {},
                    "id": results['ids'][i] if results['ids'] else None,
                }
                formatted_results.append(result_entry)
                
        # Sort by timestamp (newest first)
        formatted_results.sort(key=lambda x: x['metadata'].get('timestamp_unix', 0), reverse=True)
        return formatted_results


def get_responses_by_batch_id(
    collection: chromadb.Collection,
    batch_id: str,
    model_name: Optional[str] = None,
    query_text: Optional[str] = None,
    n_results: int = 100
) -> List[Dict[str, Any]]:
    """
    Retrieve responses from a specific batch.
    
    Args:
        collection: ChromaDB collection containing the embedded responses.
        batch_id: Batch ID to filter by.
        model_name: Optional model name to further filter results.
        query_text: Optional text to search for within the batch.
        n_results: Number of results to return.
        
    Returns:
        List of dictionaries containing the retrieved responses and metadata.
    """
    # Build filter
    batch_filter = {"batch_id": batch_id}
    
    # Add model filter if specified
    if model_name:
        batch_filter = {
            "$and": [
                {"batch_id": batch_id},
                {"model_name": model_name}
            ]
        }
    
    if query_text:
        return query_vector_database(
            collection=collection,
            query_text=query_text,
            n_results=n_results,
            filter_metadata=batch_filter
        )
    else:
        # If no query text, return all responses in the batch
        results = collection.get(
            where=batch_filter,
            limit=n_results
        )
        
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


def list_available_batches(collection: chromadb.Collection) -> List[Dict[str, Any]]:
    """
    List all available batch IDs in the collection with summary information.
    
    Args:
        collection: ChromaDB collection to query.
        
    Returns:
        List of dictionaries with batch information.
    """
    try:
        # Get all unique batch IDs
        results = collection.get()
        
        if not results or not results['metadatas']:
            return []
        
        # Extract unique batch IDs and their timestamps
        batch_info = {}
        for i, metadata in enumerate(results['metadatas']):
            batch_id = metadata.get('batch_id')
            if not batch_id:
                continue
                
            if batch_id not in batch_info:
                # Initialize new batch entry
                timestamp = metadata.get('timestamp')
                if timestamp and isinstance(timestamp, str):
                    try:
                        dt = datetime.fromisoformat(timestamp)
                    except ValueError:
                        dt = None
                else:
                    dt = None
                
                batch_info[batch_id] = {
                    'batch_id': batch_id,
                    'timestamp': timestamp,
                    'datetime': dt,
                    'models': set(),
                    'count': 0
                }
            
            # Update batch info
            batch_info[batch_id]['count'] += 1
            model = metadata.get('model_name')
            if model:
                batch_info[batch_id]['models'].add(model)
        
        # Convert to list and format
        batches = list(batch_info.values())
        for batch in batches:
            batch['models'] = list(batch['models'])
        
        # Sort by timestamp (newest first)
        batches.sort(key=lambda x: x['datetime'] or datetime.min, reverse=True)
        
        return batches
        
    except Exception as e:
        logger.error(f"Error listing batches: {str(e)}")
        return []