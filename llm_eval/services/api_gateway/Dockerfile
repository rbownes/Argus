FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install the package
RUN pip install -e .

# Expose the port
EXPOSE 8080

# Run the API Gateway
CMD ["python", "llm_eval/services/api_gateway/app.py"]
