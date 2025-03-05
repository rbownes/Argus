FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install the package
RUN pip install -e .

# Default service port
EXPOSE 8003

# Run the service (using API gateway for now, will be updated with a standalone service)
CMD ["python", "-m", "uvicorn", "services.api_gateway.app:app", "--host", "0.0.0.0", "--port", "8003"]
