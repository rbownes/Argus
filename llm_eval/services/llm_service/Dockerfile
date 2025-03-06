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

# Default service port
EXPOSE 8000

# Run the service
CMD ["python", "-m", "uvicorn", "llm_eval.services.llm_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
