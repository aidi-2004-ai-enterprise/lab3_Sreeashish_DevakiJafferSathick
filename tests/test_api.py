import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from app.main import app, model_manager

client = TestClient(app)

class TestPenguinAPI:
    """Comprehensive test suite for Penguin Classification API"""
    
    def test_predict_endpoint_valid_input(self):
        """Test prediction with valid penguin data"""
        sample_data = {
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "male",
            "island": "Torgersen"
        }
        response = client.post("/predict", json=sample_data)
        assert response.status_code == 200
        
        result = response.json()
        assert "predicted_species" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert isinstance(result["confidence"], float)
        assert 0 <= result["confidence"] <= 1
        assert result["predicted_species"] in ["Adelie", "Chinstrap", "Gentoo"]
        
        # Check probabilities sum to 1
        probs = result["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_predict_endpoint_missing_fields(self):
        """Test handling of missing required fields"""
        # Missing bill_length_mm
        incomplete_data = {
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "male",
            "island": "Torgersen"
        }
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422  # Unprocessable Entity
        
        # Missing multiple fields
        minimal_data = {
            "bill_length_mm": 39.1
        }
        response = client.post("/predict", json=minimal_data)
        assert response.status_code == 422

    def test_predict_endpoint_invalid_data_types(self):
        """Test handling of invalid data types"""
        # String instead of float
        invalid_data = {
            "bill_length_mm": "invalid_string",
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "male",
            "island": "Torgersen"
        }
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422
        
        # Boolean instead of numeric
        invalid_data2 = {
            "bill_length_mm": 39.1,
            "bill_depth_mm": True,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "male",
            "island": "Torgersen"
        }
        response = client.post("/predict", json=invalid_data2)
        assert response.status_code == 422

    def test_predict_endpoint_invalid_enum_values(self):
        """Test handling of invalid enum values"""
        # Invalid sex value
        invalid_sex_data = {
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "unknown",
            "island": "Torgersen"
        }
        response = client.post("/predict", json=invalid_sex_data)
        assert response.status_code == 422
        
        # Invalid island value
        invalid_island_data = {
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "male",
            "island": "Unknown_Island"
        }
        response = client.post("/predict", json=invalid_island_data)
        assert response.status_code == 422

    def test_predict_endpoint_out_of_range_values(self):
        """Test handling of out-of-range values"""
        # Negative values (should fail due to gt=0 constraint)
        negative_data = {
            "bill_length_mm": -10.0,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "male",
            "island": "Torgersen"
        }
        response = client.post("/predict", json=negative_data)
        assert response.status_code == 422
        
        # Extremely large values (should still work but might be unrealistic)
        extreme_data = {
            "bill_length_mm": 1000.0,
            "bill_depth_mm": 1000.0,
            "flipper_length_mm": 1000.0,
            "body_mass_g": 100000.0,
            "sex": "female",
            "island": "Biscoe"
        }
        response = client.post("/predict", json=extreme_data)
        assert response.status_code == 200  # Should work but with low confidence

    def test_predict_endpoint_boundary_conditions(self):
        """Test boundary conditions with very small positive values"""
        boundary_data = {
            "bill_length_mm": 0.1,
            "bill_depth_mm": 0.1,
            "flipper_length_mm": 0.1,
            "body_mass_g": 0.1,
            "sex": "female",
            "island": "Dream"
        }
        response = client.post("/predict", json=boundary_data)
        assert response.status_code == 200

    def test_predict_endpoint_empty_request(self):
        """Test handling of completely empty request"""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_endpoint_null_values(self):
        """Test handling of null/None values"""
        null_data = {
            "bill_length_mm": None,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "male",
            "island": "Torgersen"
        }
        response = client.post("/predict", json=null_data)
        assert response.status_code == 422

    def test_root_endpoint(self):
        """Test root endpoint returns correct information"""
        response = client.get("/")
        assert response.status_code == 200
        
        result = response.json()
        assert "message" in result
        assert "version" in result
        assert "docs" in result
        assert "prediction_endpoint" in result
        assert result["prediction_endpoint"] == "/predict"

    def test_health_check_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        result = response.json()
        assert "status" in result
        assert "model_loaded" in result
        assert result["status"] == "healthy"
        assert result["model_loaded"] is True

    def test_model_prediction_consistency(self):
        """Test that same input gives consistent predictions"""
        sample_data = {
            "bill_length_mm": 42.0,
            "bill_depth_mm": 20.0,
            "flipper_length_mm": 190.0,
            "body_mass_g": 4000.0,
            "sex": "male",
            "island": "Biscoe"
        }
        
        # Make multiple requests with same data
        responses = []
        for _ in range(3):
            response = client.post("/predict", json=sample_data)
            assert response.status_code == 200
            responses.append(response.json())
        
        # Check all responses are identical
        first_response = responses[0]
        for response in responses[1:]:
            assert response["predicted_species"] == first_response["predicted_species"]
            assert response["confidence"] == first_response["confidence"]
            assert response["probabilities"] == first_response["probabilities"]

    def test_different_species_predictions(self):
        """Test that different inputs can predict different species"""
        # Data likely to be Adelie (smaller penguin)
        adelie_data = {
            "bill_length_mm": 35.0,
            "bill_depth_mm": 19.0,
            "flipper_length_mm": 175.0,
            "body_mass_g": 3200.0,
            "sex": "female",
            "island": "Torgersen"
        }
        
        # Data likely to be Gentoo (larger penguin)
        gentoo_data = {
            "bill_length_mm": 50.0,
            "bill_depth_mm": 15.0,
            "flipper_length_mm": 220.0,
            "body_mass_g": 5500.0,
            "sex": "male",
            "island": "Biscoe"
        }
        
        adelie_response = client.post("/predict", json=adelie_data)
        gentoo_response = client.post("/predict", json=gentoo_data)
        
        assert adelie_response.status_code == 200
        assert gentoo_response.status_code == 200
        
        # Predictions might be different (though not guaranteed)
        adelie_result = adelie_response.json()
        gentoo_result = gentoo_response.json()
        
        # At least verify both are valid species
        valid_species = ["Adelie", "Chinstrap", "Gentoo"]
        assert adelie_result["predicted_species"] in valid_species
        assert gentoo_result["predicted_species"] in valid_species

    @patch('app.main.model_manager.model', None)
    def test_model_not_loaded_error(self):
        """Test behavior when model is not loaded"""
        sample_data = {
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "male",
            "island": "Torgersen"
        }
        response = client.post("/predict", json=sample_data)
        assert response.status_code == 500
        
        result = response.json()
        assert "detail" in result

    def test_model_feature_preparation(self):
        """Test that feature preparation works correctly"""
        from app.main import PenguinFeatures
        
        # Test feature preparation directly
        features = PenguinFeatures(
            bill_length_mm=39.1,
            bill_depth_mm=18.7,
            flipper_length_mm=181.0,
            body_mass_g=3750.0,
            sex="male",
            island="Torgersen"
        )
        
        prepared_features = model_manager._prepare_features(features)
        
        # Check that prepared features have correct shape and columns
        assert prepared_features.shape[0] == 1  # Single row
        assert prepared_features.shape[1] == len(model_manager.feature_columns)
        
        # Check that all expected columns are present
        for col in model_manager.feature_columns:
            assert col in prepared_features.columns

    def test_api_documentation_accessible(self):
        """Test that API documentation is accessible"""
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_spec = response.json()
        assert "openapi" in openapi_spec
        assert "info" in openapi_spec
        assert "paths" in openapi_spec
        
        # Check that our endpoints are documented
        paths = openapi_spec["paths"]
        assert "/" in paths
        assert "/health" in paths
        assert "/predict" in paths

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=app.main",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov"
    ])