"""
Tests for the API endpoints.
"""
import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import numpy as np

from src.api.app import app

client = TestClient(app)

@pytest.fixture
def test_image():
    """Create a test image for testing."""
    # Create a simple test image
    img = Image.new('RGB', (224, 224), color='white')
    # Draw a simple line to simulate a crack
    img_array = np.array(img)
    img_array[100:120, :] = [0, 0, 0]  # Black line
    img = Image.fromarray(img_array)
    
    # Save to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr

def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_predict_endpoint(test_image):
    """Test the predict endpoint."""
    response = client.post(
        "/predict",
        files={"file": ("test.png", test_image, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert isinstance(data["prediction"], str)
    assert isinstance(data["confidence"], float)

def test_metrics_endpoint():
    """Test the metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert "precision" in data
    assert "recall" in data
    assert "f1_score" in data

def test_train_endpoint():
    """Test the training endpoint."""
    response = client.post(
        "/train",
        data={
            "epochs": 2,
            "batch_size": 32,
            "learning_rate": 0.001
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "Training started"

def test_invalid_image():
    """Test prediction with invalid image."""
    # Create an invalid image file
    invalid_image = io.BytesIO(b"invalid image data")
    
    response = client.post(
        "/predict",
        files={"file": ("invalid.png", invalid_image, "image/png")}
    )
    assert response.status_code == 400

def test_missing_file():
    """Test prediction with missing file."""
    response = client.post("/predict")
    assert response.status_code == 422

def test_invalid_training_params():
    """Test training with invalid parameters."""
    response = client.post(
        "/train",
        data={
            "epochs": -1,  # Invalid epochs
            "batch_size": 0,  # Invalid batch size
            "learning_rate": -0.001  # Invalid learning rate
        }
    )
    assert response.status_code == 422

def test_websocket_connection():
    """Test WebSocket connection for training updates."""
    with client.websocket_connect("/ws/training") as websocket:
        # Start training
        client.post(
            "/train",
            data={
                "epochs": 1,
                "batch_size": 32,
                "learning_rate": 0.001
            }
        )
        
        # Receive training update
        data = websocket.receive_json()
        assert "current_epoch" in data
        assert "total_epochs" in data
        assert "loss" in data
        assert "accuracy" in data 