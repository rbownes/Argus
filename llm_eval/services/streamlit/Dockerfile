FROM python:3.9-slim

WORKDIR /app

# Copy only the requirements file first
COPY requirements.txt .

# Remove the -e . line from requirements.txt before installing
RUN sed -i '/^-e \./d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Default Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "llm_eval/services/streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
