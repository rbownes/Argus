import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(page_title="LLM Evaluation Dashboard", layout="wide")

st.title("LLM Evaluation Dashboard")

st.write("""
## Welcome to the LLM Evaluation Framework

This dashboard provides visualizations and analysis tools for comparing
the performance of different large language models across various metrics.
""")

# Connect to API Gateway
API_URL = "http://api_gateway:8080"

# Get available models
@st.cache_data(ttl=600)
def load_models():
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except:
        return []

models = load_models()
model_names = [m["name"] for m in models] if models else []

if model_names:
    st.write(f"Found {len(model_names)} available models")
else:
    st.warning("No models available. Make sure the API Gateway is running.")

# Show sample comparisons
st.header("Model Comparison")

selected_models = st.multiselect("Select models to compare", model_names, default=model_names[:2] if len(model_names) > 1 else model_names)

if selected_models:
    # This would be replaced with real data from the API
    sample_data = {
        model: {
            "toxicity": round(0.8 + 0.1 * i, 2),
            "relevance": round(0.75 + 0.05 * i, 2),
            "coherence": round(0.9 - 0.03 * i, 2)
        } for i, model in enumerate(selected_models)
    }
    
    # Convert to DataFrame for visualization
    df = pd.DataFrame([
        {"model": model, "metric": metric, "score": score}
        for model, metrics in sample_data.items()
        for metric, score in metrics.items()
    ])
    
    # Create a bar chart
    fig = px.bar(
        df,
        x="model",
        y="score",
        color="metric",
        barmode="group",
        title="Model Performance Comparison",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Show a sample response
st.header("Sample Responses")

sample_prompts = [
    "Explain quantum computing in simple terms",
    "Write a short poem about artificial intelligence",
    "What are the ethical implications of advanced AI?"
]

selected_prompt = st.selectbox("Select a prompt", sample_prompts)

if selected_prompt and selected_models:
    for model in selected_models:
        st.subheader(f"Response from {model}")
        # This would be replaced with real responses from the API
        st.write(f"This is a sample response from {model} to the prompt: {selected_prompt}")
        st.write("\n".join(["Lorem ipsum dolor sit amet"] * 3))
        
        # Show metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Toxicity", f"{sample_data[model]['toxicity'] * 100:.1f}%")
        col2.metric("Relevance", f"{sample_data[model]['relevance'] * 100:.1f}%")
        col3.metric("Coherence", f"{sample_data[model]['coherence'] * 100:.1f}%")
        
        st.divider()
