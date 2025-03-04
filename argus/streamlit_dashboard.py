"""
Interactive Streamlit dashboard for LLM analysis.

This interactive dashboard provides visualizations and analysis tools
for exploring LLM responses, metrics, and comparisons.

Run with: streamlit run streamlit_dashboard.py
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional, Tuple

# Try to import ChromaDB
try:
    import chromadb
except ImportError:
    chromadb = None

# Try to import nltk for text analysis
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.probability import FreqDist
    # Download necessary resources
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="LLM Analytics Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
DEFAULT_SQLITE_PATH = "./llm_data/llm_data.sqlite"
DEFAULT_CHROMA_PATH = "./vector_db"

# Helper functions
def load_sqlite_connection(db_path: str) -> sqlite3.Connection:
    """Load SQLite connection."""
    if not os.path.exists(db_path):
        st.error(f"SQLite database not found at: {db_path}")
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except Exception as e:
        st.error(f"Error connecting to SQLite database: {e}")
        return None

def load_chroma_connection(chroma_path: str) -> Optional[chromadb.PersistentClient]:
    """Load ChromaDB connection."""
    if not chromadb:
        st.warning("ChromaDB not installed. Install with: pip install chromadb")
        return None
    
    if not os.path.exists(chroma_path):
        st.error(f"ChromaDB directory not found at: {chroma_path}")
        return None
    
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        return client
    except Exception as e:
        st.error(f"Error connecting to ChromaDB: {e}")
        return None

def get_collections(chroma_client: chromadb.PersistentClient) -> List[str]:
    """Get list of collections in ChromaDB."""
    if not chroma_client:
        return []
    
    try:
        collections = chroma_client.list_collections()
        return [c.name for c in collections]
    except Exception as e:
        st.error(f"Error listing ChromaDB collections: {e}")
        return []

def get_batches(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get batches from SQLite."""
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT 
            id, timestamp, description,
            (SELECT COUNT(*) FROM responses WHERE batch_id = batches.id) as response_count
        FROM batches
        ORDER BY timestamp DESC
        """
        return pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"Error fetching batches: {e}")
        return pd.DataFrame()

def get_models(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get models from SQLite."""
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT 
            id, name, provider, version, first_used, last_used,
            (SELECT COUNT(*) FROM responses WHERE model_id = models.id) as response_count
        FROM models
        ORDER BY last_used DESC
        """
        return pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return pd.DataFrame()

def get_metrics(conn: sqlite3.Connection, batch_id: Optional[str] = None) -> pd.DataFrame:
    """Get metrics from SQLite."""
    if not conn:
        return pd.DataFrame()
    
    try:
        if batch_id:
            query = """
            SELECT m.id, m.name, m.batch_id, mo.name as model_name, 
                   m.timestamp, m.value, m.sample_count, m.success, 
                   m.error, m.metadata, m.details
            FROM metrics m
            LEFT JOIN models mo ON m.model_id = mo.id
            WHERE m.batch_id = ?
            ORDER BY m.timestamp DESC
            """
            return pd.read_sql_query(query, conn, params=(batch_id,))
        else:
            query = """
            SELECT m.id, m.name, m.batch_id, mo.name as model_name, 
                   m.timestamp, m.value, m.sample_count, m.success, 
                   m.error, m.metadata, m.details
            FROM metrics m
            LEFT JOIN models mo ON m.model_id = mo.id
            ORDER BY m.timestamp DESC
            LIMIT 1000
            """
            return pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"Error fetching metrics: {e}")
        return pd.DataFrame()

def get_batch_responses(conn: sqlite3.Connection, batch_id: str) -> pd.DataFrame:
    """Get responses for a specific batch."""
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT 
            r.id, m.name as model_name, p.text as prompt, p.prompt_idx,
            r.text as response, r.timestamp, r.embedding_id, r.metadata
        FROM responses r
        JOIN models m ON r.model_id = m.id
        JOIN prompts p ON r.prompt_id = p.id
        WHERE r.batch_id = ?
        ORDER BY p.prompt_idx, m.name
        """
        return pd.read_sql_query(query, conn, params=(batch_id,))
    except Exception as e:
        st.error(f"Error fetching batch responses: {e}")
        return pd.DataFrame()

def process_metrics_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Process metrics dataframe for display."""
    if df.empty:
        return df
    
    # Convert JSON columns
    for col in ['metadata', 'details']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else {})
    
    return df

def analyze_text(text: str) -> Dict[str, Any]:
    """Analyze text for dashboard display."""
    if not NLTK_AVAILABLE:
        return {
            "word_count": len(text.split()),
            "char_count": len(text)
        }
    
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Filter tokens
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    
    # Calculate frequency distribution
    fdist = FreqDist(filtered_tokens)
    
    return {
        "word_count": len(tokens),
        "char_count": len(text),
        "unique_words": len(set(filtered_tokens)),
        "lexical_diversity": len(set(filtered_tokens)) / len(filtered_tokens) if filtered_tokens else 0,
        "top_words": dict(fdist.most_common(10))
    }

def compare_responses(responses_df: pd.DataFrame) -> None:
    """Create visualizations comparing responses across models."""
    if responses_df.empty:
        st.info("No responses to compare.")
        return
    
    # Group by prompt and model
    grouped = responses_df.groupby(['prompt_idx', 'model_name'])
    
    # Get unique prompts and models
    unique_prompts = responses_df['prompt_idx'].unique()
    unique_models = responses_df['model_name'].unique()
    
    # Create metrics for comparison
    response_lengths = responses_df.groupby('model_name')['response'].apply(lambda x: x.str.len().mean()).reset_index()
    response_lengths.columns = ['Model', 'Average Response Length']
    
    # Length comparison
    st.subheader("Response Length Comparison")
    fig = px.bar(
        response_lengths, 
        x='Model', 
        y='Average Response Length',
        color='Model',
        title="Average Response Length by Model"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Word count comparison if NLTK is available
    if NLTK_AVAILABLE:
        # Calculate word counts
        responses_df['word_count'] = responses_df['response'].apply(lambda x: len(word_tokenize(x)))
        word_counts = responses_df.groupby('model_name')['word_count'].mean().reset_index()
        word_counts.columns = ['Model', 'Average Word Count']
        
        fig = px.bar(
            word_counts, 
            x='Model', 
            y='Average Word Count',
            color='Model',
            title="Average Word Count by Model"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate lexical diversity
        def calc_lexical_diversity(text):
            tokens = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
            return len(set(filtered_tokens)) / len(filtered_tokens) if filtered_tokens else 0
        
        responses_df['lexical_diversity'] = responses_df['response'].apply(calc_lexical_diversity)
        diversity = responses_df.groupby('model_name')['lexical_diversity'].mean().reset_index()
        diversity.columns = ['Model', 'Average Lexical Diversity']
        
        fig = px.bar(
            diversity, 
            x='Model', 
            y='Average Lexical Diversity',
            color='Model',
            title="Average Lexical Diversity by Model"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Select a prompt for detailed comparison
    st.subheader("Detailed Response Comparison")
    selected_prompt_idx = st.selectbox(
        "Select a prompt to compare responses:",
        options=sorted(unique_prompts)
    )
    
    # Get the prompt text
    prompt_text = responses_df[responses_df['prompt_idx'] == selected_prompt_idx]['prompt'].iloc[0]
    st.markdown("**Prompt:**")
    st.markdown(f"> {prompt_text}")
    
    # Show responses side by side
    st.markdown("**Responses:**")
    
    # Get responses for the selected prompt
    prompt_responses = responses_df[responses_df['prompt_idx'] == selected_prompt_idx].sort_values('model_name')
    
    # Create columns for each model
    cols = st.columns(len(unique_models))
    
    for i, model in enumerate(sorted(unique_models)):
        model_response = prompt_responses[prompt_responses['model_name'] == model]
        
        if not model_response.empty:
            response_text = model_response['response'].iloc[0]
            
            # Analyze response
            analysis = analyze_text(response_text)
            
            cols[i].markdown(f"**{model}**")
            cols[i].markdown(f"{response_text[:500]}..." if len(response_text) > 500 else response_text)
            cols[i].markdown("---")
            cols[i].markdown(f"Word count: {analysis['word_count']}")
            cols[i].markdown(f"Character count: {analysis['char_count']}")
            
            if NLTK_AVAILABLE:
                cols[i].markdown(f"Unique words: {analysis['unique_words']}")
                cols[i].markdown(f"Lexical diversity: {analysis['lexical_diversity']:.3f}")
                
                # Top words
                if analysis['top_words']:
                    cols[i].markdown("**Top words:**")
                    top_words_df = pd.DataFrame(
                        list(analysis['top_words'].items()),
                        columns=['Word', 'Frequency']
                    )
                    cols[i].dataframe(top_words_df, hide_index=True)

def display_metrics_comparison(metrics_df: pd.DataFrame) -> None:
    """Display metrics comparison visualizations."""
    if metrics_df.empty:
        st.info("No metrics data available.")
        return
    
    # Process metrics dataframe
    metrics_df = process_metrics_for_display(metrics_df)
    
    # Get unique metric names
    metric_names = metrics_df['name'].unique()
    
    # Select metric to visualize
    selected_metric = st.selectbox(
        "Select metric to visualize:",
        options=metric_names
    )
    
    # Filter for selected metric
    metric_data = metrics_df[metrics_df['name'] == selected_metric]
    
    # Check if we have model-specific data
    has_model_data = metric_data['model_name'].notna().any()
    
    if has_model_data:
        # Group by model
        model_metrics = metric_data.dropna(subset=['model_name']).groupby('model_name')['value'].mean().reset_index()
        model_metrics.columns = ['Model', 'Average Value']
        
        # Create bar chart
        fig = px.bar(
            model_metrics,
            x='Model',
            y='Average Value',
            color='Model',
            title=f"Average {selected_metric} by Model"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Metric over time (if timestamps are available)
    if 'timestamp' in metric_data.columns:
        metric_data['timestamp'] = pd.to_datetime(metric_data['timestamp'])
        metric_data = metric_data.sort_values('timestamp')
        
        if has_model_data:
            # Line chart by model
            fig = px.line(
                metric_data.dropna(subset=['model_name']),
                x='timestamp',
                y='value',
                color='model_name',
                title=f"{selected_metric} Over Time by Model"
            )
            fig.update_layout(xaxis_title="Time", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Simple line chart
            fig = px.line(
                metric_data,
                x='timestamp',
                y='value',
                title=f"{selected_metric} Over Time"
            )
            fig.update_layout(xaxis_title="Time", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)
    
    # Distribution of values
    st.subheader(f"Distribution of {selected_metric} Values")
    fig = px.histogram(
        metric_data,
        x='value',
        color='model_name' if has_model_data else None,
        nbins=20,
        title=f"Distribution of {selected_metric} Values"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Raw data
    with st.expander("View Raw Metric Data"):
        st.dataframe(metric_data)

def display_semantic_search(chroma_client: chromadb.PersistentClient, collection_name: str) -> None:
    """Display semantic search interface."""
    if not chroma_client:
        st.warning("ChromaDB connection not available.")
        return
    
    st.subheader("Semantic Search")
    
    try:
        # Get collection
        collection = chroma_client.get_collection(collection_name)
        
        # Search interface
        search_query = st.text_input("Enter search query:")
        search_filter = st.text_input("Filter (optional, JSON format):", "{}")
        n_results = st.slider("Number of results:", 1, 20, 5)
        
        if st.button("Search"):
            if not search_query:
                st.warning("Please enter a search query.")
                return
            
            # Parse filter if provided
            filter_dict = None
            if search_filter and search_filter != "{}":
                try:
                    filter_dict = json.loads(search_filter)
                except json.JSONDecodeError:
                    st.error("Invalid JSON filter. Please check the format.")
                    return
            
            # Perform search
            results = collection.query(
                query_texts=[search_query],
                n_results=n_results,
                where=filter_dict
            )
            
            # Display results
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else None
                    
                    st.markdown(f"**Result {i+1}** (Similarity: {1-distance:.4f})")
                    
                    # Display metadata
                    if metadata:
                        model = metadata.get('model_name', 'Unknown Model')
                        prompt = metadata.get('prompt', 'No prompt available')
                        
                        st.markdown(f"**Model:** {model}")
                        st.markdown("**Prompt:**")
                        st.markdown(f"> {prompt}")
                    
                    # Display document
                    st.markdown("**Response:**")
                    st.markdown(f"{doc[:500]}..." if len(doc) > 500 else doc)
                    st.markdown("---")
            else:
                st.info("No results found.")
                
    except Exception as e:
        st.error(f"Error performing semantic search: {e}")

def main():
    # Sidebar for configuration
    st.sidebar.title("LLM Analytics Dashboard")
    
    # Database connections
    st.sidebar.header("Data Sources")
    
    sqlite_path = st.sidebar.text_input("SQLite Database Path:", DEFAULT_SQLITE_PATH)
    chroma_path = st.sidebar.text_input("ChromaDB Path:", DEFAULT_CHROMA_PATH)
    
    # Connect to databases
    sqlite_conn = load_sqlite_connection(sqlite_path)
    chroma_client = load_chroma_connection(chroma_path)
    
    # Main content
    st.title("LLM Analytics Dashboard")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Batch Analysis", "Model Comparison", "Metrics Analysis", "Semantic Search"]
    )
    
    # Overview page
    if page == "Overview":
        st.header("System Overview")
        
        # Check connections
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("SQLite Status")
            if sqlite_conn:
                st.success("Connected")
                
                # Get stats
                batches_df = get_batches(sqlite_conn)
                models_df = get_models(sqlite_conn)
                
                st.metric("Total Batches", len(batches_df))
                st.metric("Total Models", len(models_df))
                
                if not batches_df.empty and 'response_count' in batches_df.columns:
                    total_responses = batches_df['response_count'].sum()
                    st.metric("Total Responses", total_responses)
            else:
                st.error("Not Connected")
        
        with col2:
            st.subheader("ChromaDB Status")
            if chroma_client:
                st.success("Connected")
                
                # Get collections
                collections = get_collections(chroma_client)
                
                st.metric("Total Collections", len(collections))
                
                if collections:
                    st.write("Collections:")
                    for collection in collections:
                        st.write(f"- {collection}")
            else:
                st.error("Not Connected")
        
        # Display batch history
        if sqlite_conn:
            st.subheader("Recent Batches")
            batches_df = get_batches(sqlite_conn)
            
            if not batches_df.empty:
                batches_df['timestamp'] = pd.to_datetime(batches_df['timestamp'])
                
                st.dataframe(
                    batches_df,
                    hide_index=True,
                    column_config={
                        "id": "Batch ID",
                        "timestamp": st.column_config.DatetimeColumn("Timestamp"),
                        "description": "Description",
                        "response_count": st.column_config.NumberColumn("Responses")
                    }
                )
                
                # Plot batches over time
                if len(batches_df) > 1:
                    st.subheader("Batch Activity")
                    batches_df = batches_df.sort_values('timestamp')
                    
                    # Group by date
                    batches_df['date'] = batches_df['timestamp'].dt.date
                    daily_counts = batches_df.groupby('date').size().reset_index()
                    daily_counts.columns = ['Date', 'Batch Count']
                    
                    fig = px.bar(
                        daily_counts,
                        x='Date',
                        y='Batch Count',
                        title="Batches per Day"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No batches found in the database.")
        
        # Display model info
        if sqlite_conn:
            st.subheader("Models")
            models_df = get_models(sqlite_conn)
            
            if not models_df.empty:
                # Convert dates
                for col in ['first_used', 'last_used']:
                    if col in models_df.columns:
                        models_df[col] = pd.to_datetime(models_df[col])
                
                st.dataframe(
                    models_df,
                    hide_index=True,
                    column_config={
                        "name": "Model Name",
                        "provider": "Provider",
                        "version": "Version",
                        "first_used": st.column_config.DatetimeColumn("First Used"),
                        "last_used": st.column_config.DatetimeColumn("Last Used"),
                        "response_count": st.column_config.NumberColumn("Response Count")
                    }
                )
                
                # Model usage pie chart
                if 'response_count' in models_df.columns:
                    st.subheader("Model Usage")
                    
                    fig = px.pie(
                        models_df,
                        values='response_count',
                        names='name',
                        title="Response Distribution by Model"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No models found in the database.")
    
    # Batch Analysis page
    elif page == "Batch Analysis":
        st.header("Batch Analysis")
        
        if not sqlite_conn:
            st.warning("SQLite connection required for batch analysis.")
            return
        
        # Get batches
        batches_df = get_batches(sqlite_conn)
        
        if batches_df.empty:
            st.info("No batches found in the database.")
            return
        
        # Batch selector
        selected_batch_id = st.selectbox(
            "Select a batch to analyze:",
            options=batches_df['id'].tolist(),
            format_func=lambda x: f"{x} ({batches_df[batches_df['id'] == x]['timestamp'].iloc[0]})"
        )
        
        if selected_batch_id:
            # Get batch info
            batch_info = batches_df[batches_df['id'] == selected_batch_id].iloc[0]
            
            # Display batch info
            st.subheader("Batch Information")
            col1, col2, col3 = st.columns(3)
            col1.metric("Batch ID", selected_batch_id)
            col2.metric("Timestamp", pd.to_datetime(batch_info['timestamp']).strftime('%Y-%m-%d %H:%M:%S'))
            col3.metric("Response Count", batch_info['response_count'])
            
            if 'description' in batch_info and batch_info['description']:
                st.markdown(f"**Description:** {batch_info['description']}")
            
            # Get responses for this batch
            responses_df = get_batch_responses(sqlite_conn, selected_batch_id)
            
            if responses_df.empty:
                st.info("No responses found for this batch.")
                return
            
            # Analyze and display responses
            st.subheader("Responses Analysis")
            
            # Summary by model
            model_counts = responses_df['model_name'].value_counts().reset_index()
            model_counts.columns = ['Model', 'Response Count']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Response Count by Model**")
                fig = px.bar(
                    model_counts,
                    x='Model',
                    y='Response Count',
                    color='Model'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Response Length by Model**")
                responses_df['response_length'] = responses_df['response'].str.len()
                length_by_model = responses_df.groupby('model_name')['response_length'].mean().reset_index()
                length_by_model.columns = ['Model', 'Average Length']
                
                fig = px.bar(
                    length_by_model,
                    x='Model',
                    y='Average Length',
                    color='Model'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Metrics for this batch
            metrics_df = get_metrics(sqlite_conn, selected_batch_id)
            
            if not metrics_df.empty:
                st.subheader("Batch Metrics")
                
                # Process metrics for display
                metrics_df = process_metrics_for_display(metrics_df)
                
                # Get unique metrics
                unique_metrics = metrics_df['name'].unique()
                
                metric_tabs = st.tabs(list(unique_metrics))
                
                for i, metric_name in enumerate(unique_metrics):
                    with metric_tabs[i]:
                        metric_data = metrics_df[metrics_df['name'] == metric_name]
                        
                        # Display by model if available
                        if 'model_name' in metric_data.columns and metric_data['model_name'].notna().any():
                            model_metrics = metric_data.dropna(subset=['model_name']).groupby('model_name')['value'].mean().reset_index()
                            model_metrics.columns = ['Model', 'Average Value']
                            
                            st.markdown(f"**{metric_name} by Model**")
                            
                            # Bar chart
                            fig = px.bar(
                                model_metrics,
                                x='Model',
                                y='Average Value',
                                color='Model'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Single value
                            st.metric(metric_name, f"{metric_data['value'].mean():.4f}")
                        
                        # Show details if available
                        if 'details' in metric_data.columns:
                            details = metric_data['details'].iloc[0]
                            if details and isinstance(details, dict):
                                with st.expander("View Metric Details"):
                                    st.json(details)
            
            # Compare responses
            st.subheader("Response Comparison")
            compare_responses(responses_df)
            
            # Raw responses
            with st.expander("View Raw Responses"):
                st.dataframe(
                    responses_df,
                    column_config={
                        "model_name": "Model",
                        "prompt": "Prompt",
                        "response": "Response",
                        "timestamp": "Timestamp"
                    }
                )
    
    # Model Comparison page
    elif page == "Model Comparison":
        st.header("Model Comparison")
        
        if not sqlite_conn:
            st.warning("SQLite connection required for model comparison.")
            return
        
        # Get models
        models_df = get_models(sqlite_conn)
        
        if models_df.empty:
            st.info("No models found in the database.")
            return
        
        # Model selector
        selected_models = st.multiselect(
            "Select models to compare:",
            options=models_df['name'].tolist(),
            default=models_df['name'].tolist()[:2]  # Default to first two models
        )
        
        if not selected_models:
            st.info("Please select at least one model to analyze.")
            return
        
        if len(selected_models) == 1:
            st.info("Select at least one more model for comparison.")
        
        # Model statistics
        st.subheader("Model Statistics")
        
        model_stats = models_df[models_df['name'].isin(selected_models)]
        
        # Display model info
        for index, row in model_stats.iterrows():
            with st.expander(f"Model: {row['name']}"):
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Provider", row['provider'] if not pd.isna(row['provider']) else "Unknown")
                col1.metric("Version", row['version'] if not pd.isna(row['version']) else "Unknown")
                
                col2.metric("First Used", pd.to_datetime(row['first_used']).strftime('%Y-%m-%d') if not pd.isna(row['first_used']) else "Unknown")
                col2.metric("Last Used", pd.to_datetime(row['last_used']).strftime('%Y-%m-%d') if not pd.isna(row['last_used']) else "Unknown")
                
                col3.metric("Response Count", row['response_count'])
        
        # Get metrics by model
        metrics_df = get_metrics(sqlite_conn)
        metrics_df = metrics_df[metrics_df['model_name'].isin(selected_models)]
        
        if not metrics_df.empty:
            st.subheader("Metrics Comparison")
            display_metrics_comparison(metrics_df)
        
        # Response comparison
        st.subheader("Response Examples")
        
        # Get batches
        batches_df = get_batches(sqlite_conn)
        
        if not batches_df.empty:
            # Batch selector
            selected_batch_id = st.selectbox(
                "Select a batch to compare responses:",
                options=batches_df['id'].tolist(),
                format_func=lambda x: f"{x} ({batches_df[batches_df['id'] == x]['timestamp'].iloc[0]})"
            )
            
            if selected_batch_id:
                # Get responses
                responses_df = get_batch_responses(sqlite_conn, selected_batch_id)
                
                # Filter to selected models
                responses_df = responses_df[responses_df['model_name'].isin(selected_models)]
                
                if not responses_df.empty:
                    compare_responses(responses_df)
                else:
                    st.info("No responses from the selected models in this batch.")
        else:
            st.info("No batches found for response comparison.")
    
    # Metrics Analysis page
    elif page == "Metrics Analysis":
        st.header("Metrics Analysis")
        
        if not sqlite_conn:
            st.warning("SQLite connection required for metrics analysis.")
            return
        
        # Get all metrics
        metrics_df = get_metrics(sqlite_conn)
        
        if metrics_df.empty:
            st.info("No metrics found in the database.")
            return
        
        # Metrics overview
        st.subheader("Metrics Overview")
        
        # Unique metric names
        metric_names = metrics_df['name'].unique()
        
        # Count by metric name
        metric_counts = metrics_df['name'].value_counts().reset_index()
        metric_counts.columns = ['Metric', 'Count']
        
        fig = px.bar(
            metric_counts,
            x='Metric',
            y='Count',
            color='Metric',
            title="Metric Counts"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed analysis
        st.subheader("Detailed Metrics Analysis")
        display_metrics_comparison(metrics_df)
    
    # Semantic Search page
    elif page == "Semantic Search":
        st.header("Semantic Search")
        
        if not chroma_client:
            st.warning("ChromaDB connection required for semantic search.")
            return
        
        # Get collections
        collections = get_collections(chroma_client)
        
        if not collections:
            st.info("No collections found in ChromaDB.")
            return
        
        # Collection selector
        selected_collection = st.selectbox(
            "Select a collection to search:",
            options=collections
        )
        
        if selected_collection:
            display_semantic_search(chroma_client, selected_collection)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### About")
    st.sidebar.info(
        "This dashboard provides analytics and visualizations for LLM outputs and metrics. "
        "It connects to SQLite for structured data and ChromaDB for vector embeddings."
    )
    
    # Close connections
    if sqlite_conn:
        sqlite_conn.close()

if __name__ == "__main__":
    main()