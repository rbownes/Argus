"""
Setup script for the LLM Evaluation Framework.
"""
from setuptools import setup, find_packages

setup(
    name="llm_eval",
    version="0.1.0",
    description="Framework for evaluating large language models",
    author="LLM Eval Team",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "litellm>=1.0.0",
        "sentence-transformers>=2.2.0",
        "asyncpg>=0.28.0",
        "chromadb>=0.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "visualization": [
            "streamlit>=1.28.0",
            "plotly>=5.17.0",
            "pandas>=2.0.0",
        ],
    },
    python_requires=">=3.9",
)
