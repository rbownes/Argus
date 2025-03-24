"""
Tests for shared utilities.
"""
import os
import pytest
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from shared.utils import (
    ApiResponse, 
    ResponseStatus, 
    ApiError, 
    get_env_var, 
    create_api_app,
    PaginationParams,
    paginate_results
)

class TestApiResponse:
    """Tests for ApiResponse model."""
    
    def test_api_response_creation(self):
        """Test creating an API response."""
        response = ApiResponse(
            status=ResponseStatus.SUCCESS,
            message="Test message",
            data={"test": "data"}
        )
        
        assert response.status == ResponseStatus.SUCCESS
        assert response.message == "Test message"
        assert response.data == {"test": "data"}
        assert isinstance(response.timestamp, str)
        
    def test_api_response_dict(self):
        """Test converting ApiResponse to dict."""
        response = ApiResponse(
            status=ResponseStatus.SUCCESS,
            message="Test message",
            data={"test": "data"}
        )
        
        response_dict = response.dict()
        assert response_dict["status"] == "success"
        assert response_dict["message"] == "Test message"
        assert response_dict["data"] == {"test": "data"}
        assert "timestamp" in response_dict

class TestApiError:
    """Tests for ApiError exception."""
    
    def test_api_error_creation(self):
        """Test creating an API error."""
        error = ApiError(
            status_code=404,
            message="Not found",
            details={"resource": "user", "id": 123}
        )
        
        assert error.status_code == 404
        assert error.message == "Not found"
        assert error.details == {"resource": "user", "id": 123}
        
    def test_api_error_as_exception(self):
        """Test using ApiError as an exception."""
        with pytest.raises(ApiError) as excinfo:
            raise ApiError(status_code=500, message="Server error")
        
        assert excinfo.value.status_code == 500
        assert excinfo.value.message == "Server error"

class TestEnvironmentUtils:
    """Tests for environment utilities."""
    
    def test_get_env_var_with_default(self):
        """Test getting environment variable with default."""
        # Ensure the test environment variable doesn't exist
        if "TEST_VAR" in os.environ:
            del os.environ["TEST_VAR"]
            
        value = get_env_var("TEST_VAR", default="default_value")
        assert value == "default_value"
        
    def test_get_env_var_with_value(self):
        """Test getting environment variable with a value."""
        os.environ["TEST_VAR"] = "test_value"
        value = get_env_var("TEST_VAR", default="default_value")
        assert value == "test_value"
        
    def test_get_env_var_required_missing(self):
        """Test getting required environment variable that's missing."""
        if "TEST_VAR" in os.environ:
            del os.environ["TEST_VAR"]
            
        with pytest.raises(ValueError) as excinfo:
            get_env_var("TEST_VAR", required=True)
        
        assert "Required environment variable TEST_VAR is not set" in str(excinfo.value)

class TestApiApp:
    """Tests for API app creation."""
    
    def test_create_api_app(self):
        """Test creating an API app."""
        app = create_api_app(
            title="Test API",
            description="Test API description",
            version="0.1.0"
        )
        
        assert isinstance(app, FastAPI)
        assert app.title == "Test API"
        assert app.description == "Test API description"
        assert app.version == "0.1.0"
        
    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        app = create_api_app(
            title="Test API",
            description="Test API description"
        )
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["message"] == "Service is healthy"
        
    def test_generic_error_handler(self):
        """Test generic error handler."""
        app = create_api_app(
            title="Test API",
            description="Test API description"
        )
        
        @app.get("/test-error")
        async def test_error():
            raise ValueError("Test error")
        
        client = TestClient(app)
        response = client.get("/test-error")
        
        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "error"
        assert data["message"] == "An unexpected error occurred"

class TestPagination:
    """Tests for pagination utilities."""
    
    def test_pagination_params(self):
        """Test pagination parameters."""
        params = PaginationParams(page=2, limit=10)
        assert params.page == 2
        assert params.limit == 10
        assert params.get_skip() == 10
        
    def test_pagination_params_defaults(self):
        """Test pagination parameters defaults."""
        params = PaginationParams()
        assert params.page == 1
        assert params.limit == 10
        assert params.get_skip() == 0
        
    def test_paginate_results(self):
        """Test paginating results."""
        params = PaginationParams(page=2, limit=10)
        items = [{"id": i} for i in range(10)]
        total_count = 25
        
        result = paginate_results(items, params, total_count)
        
        assert result["items"] == items
        assert result["pagination"]["page"] == 2
        assert result["pagination"]["limit"] == 10
        assert result["pagination"]["total"] == 25
        assert result["pagination"]["pages"] == 3
