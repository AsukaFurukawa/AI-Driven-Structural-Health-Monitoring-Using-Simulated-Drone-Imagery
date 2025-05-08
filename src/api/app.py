"""
API server for structural health monitoring.

This module provides a FastAPI server to serve the trained model for crack detection.
"""
import os
import sys
import logging
import yaml
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import torch
import numpy as np
from PIL import Image
import io
import base64
import uuid
import cv2

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, Response, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torchvision.transforms as transforms

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.crack_detection_model import get_model
from src.models.trainer import Trainer
from src.data.dataset import create_dataloaders
from src.data.drone_simulator import simulate_drone_image

# Attempt to import the model - this may fail if TensorFlow is not installed
try:
    from src.ai.models.defect_detector import DefectDetector
except ImportError:
    print("Warning: Could not import DefectDetector. Model functionality will be limited.")
    DefectDetector = None

# Initialize FastAPI app
app = FastAPI(
    title="Structural Health Monitoring API",
    description="API for analyzing infrastructure defects using AI and drone imagery",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
UPLOAD_DIR = BASE_DIR / "uploads" / "api_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Create directory for static images
STATIC_IMG_DIR = BASE_DIR / "src" / "api" / "static" / "img"
os.makedirs(STATIC_IMG_DIR, exist_ok=True)

# Load configuration
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {
        "model": {
            "name": "cnn",
            "num_classes": 2,
            "class_names": ["negative", "positive"]
        }
    }

# Templates
templates_dir = BASE_DIR / "src" / "api" / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Static files
static_dir = BASE_DIR / "src" / "api" / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Model instance
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = CONFIG.get("model", {}).get("class_names", ["negative", "positive"])

# WebSocket connections
active_connections: List[WebSocket] = []

# Sample data for dashboard
SAMPLE_DATA = {
    "total_structures": 25,
    "healthy_structures": 18,
    "healthy_percentage": 72,
    "warning_structures": 5, 
    "critical_structures": 2,
    "latest_inspection": {
        "id": "insp-20250312-001",
        "structure_name": "Howrah Bridge",
        "date": "March 12, 2025",
        "status": "Warning",
        "status_color": "warning",
        "defects_count": 3
    },
    "alerts": [
        {
            "id": "alt-001",
            "title": "Critical crack detected",
            "description": "A 25cm crack was detected on the main support beam",
            "severity": "Critical",
            "severity_color": "danger",
            "structure_name": "Howrah Bridge",
            "time_ago": "2 hours ago"
        },
        {
            "id": "alt-002",
            "title": "Corrosion detected",
            "description": "Early signs of corrosion detected on steel reinforcement",
            "severity": "Warning",
            "severity_color": "warning",
            "structure_name": "Bandra-Worli Sea Link",
            "time_ago": "Yesterday"
        },
        {
            "id": "alt-003",
            "title": "Inspection scheduled",
            "description": "Routine inspection scheduled for tomorrow",
            "severity": "Info",
            "severity_color": "info",
            "structure_name": "Bhakra Dam",
            "time_ago": "3 days ago"
        }
    ],
    "recent_inspections": [
        {
            "structure_name": "Howrah Bridge",
            "date": "March 12, 2025",
            "description": "Routine inspection detected minor cracks",
            "status": "Warning",
            "status_color": "warning"
        },
        {
            "structure_name": "Tehri Dam",
            "date": "March 10, 2025",
            "description": "Detailed inspection completed",
            "status": "Healthy",
            "status_color": "success"
        },
        {
            "structure_name": "Taj Mahal",
            "date": "March 5, 2025",
            "description": "Structural integrity verified",
            "status": "Healthy",
            "status_color": "success"
        }
    ],
    "structures": [
        {"id": 1, "name": "Howrah Bridge"},
        {"id": 2, "name": "Bandra-Worli Sea Link"},
        {"id": 3, "name": "Bhakra Dam"},
        {"id": 4, "name": "Delhi Metro Bridge"},
        {"id": 5, "name": "Taj Mahal"}
    ]
}

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model
    try:
        if DefectDetector:
            model = DefectDetector(num_classes=4)
            # In production, load pre-trained weights here
            # model.load_state_dict(torch.load("path_to_weights.pth"))
            model.eval()
            logger.info("Model loaded successfully")
        else:
            logger.warning("DefectDetector not available. Running in limited mode.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

async def broadcast_training_update(data: Dict[str, Any]):
    """Broadcast training update to all connected clients."""
    for connection in active_connections:
        try:
            await connection.send_json(data)
        except WebSocketDisconnect:
            active_connections.remove(connection)
        except Exception as e:
            logger.error(f"Error broadcasting to client: {str(e)}")

@app.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
    """WebSocket endpoint for training updates."""
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)

@app.post("/train")
async def start_training(
    epochs: int = Form(50),
    batch_size: int = Form(32),
    learning_rate: float = Form(0.001)
):
    """Start model training."""
    try:
        # Create dataloaders
        dataloaders = create_dataloaders(
            data_dir=str(BASE_DIR / "data" / "processed"),
            batch_size=batch_size,
            num_workers=4
        )
        
        # Create model
        model = get_model(
            model_name=CONFIG.get("model", {}).get("name", "cnn"),
            num_classes=CONFIG.get("model", {}).get("num_classes", 2),
            pretrained=True
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            device=device,
            config={
                "num_epochs": epochs,
                "early_stopping_patience": 10
            },
            checkpoint_dir=str(CHECKPOINT_DIR)
        )
        
        # Start training in background
        asyncio.create_task(train_with_updates(trainer, epochs))
        
        return {"status": "Training started"}
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def train_with_updates(trainer: Trainer, total_epochs: int):
    """Train model and broadcast updates."""
    try:
        for epoch in range(total_epochs):
            # Train one epoch
            train_loss, train_acc = trainer.train_one_epoch(epoch)
            
            # Validate
            val_loss, val_acc = trainer.validate(epoch)
            
            # Broadcast update
            await broadcast_training_update({
                "current_epoch": epoch + 1,
                "total_epochs": total_epochs,
                "loss": train_loss,
                "accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        await broadcast_training_update({
            "error": str(e)
        })

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess image for model inference.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Convert grayscale images to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0).to(device)

def predict(image: Image.Image) -> Dict:
    """
    Run inference on image.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        dict: Prediction results
    """
    global model
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Preprocess image
    img_tensor = preprocess_image(image)
    
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Get prediction
        _, predicted = torch.max(outputs, 1)
        predicted_idx = predicted.item()
        
        # Get class name and probability
        predicted_class = class_names[predicted_idx] if predicted_idx < len(class_names) else f"Class {predicted_idx}"
        probability = probabilities[predicted_idx].item()
        
        # Format as percentage
        probability_pct = f"{probability * 100:.2f}%"
        
        # Get all class probabilities
        all_probabilities = {
            class_names[i] if i < len(class_names) else f"Class {i}": float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        return {
            "class": predicted_class,
            "confidence": probability,
            "confidence_pct": probability_pct,
            "has_defect": predicted_idx > 0,  # Assuming class 0 is "no defect"
            "all_probabilities": all_probabilities
        }

def save_upload(image: Image.Image) -> str:
    """
    Save uploaded image to disk.
    
    Args:
        image (PIL.Image): Image to save
        
    Returns:
        str: Path to saved image
    """
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"upload_{timestamp}_{uuid.uuid4().hex[:8]}.png"
    
    # Save image
    save_path = UPLOAD_DIR / filename
    image.save(save_path)
    
    return str(save_path)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the home page."""
    # Get data from services for consistent UI
    statistics = get_statistics()
    
    # Build template context
    data = {
        "request": request,
        "current_date": datetime.now().strftime("%B %d, %Y"),
        **statistics,
        **SAMPLE_DATA
    }
    return templates.TemplateResponse("home.html", data)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the dashboard page."""
    # In a real application, this data would come from a database or other source
    data = {
        "request": request,
        "current_date": datetime.now().strftime("%B %d, %Y"),
        **SAMPLE_DATA
    }
    return templates.TemplateResponse("dashboard.html", data)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict defects in uploaded image.
    
    Args:
        file (UploadFile): Uploaded image file
        
    Returns:
        dict: Prediction results
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Save upload
        save_path = save_upload(image)
        
        # Run prediction
        result = predict(image)
        
        # Add metadata
        result["timestamp"] = datetime.now().isoformat()
        result["filename"] = file.filename
        result["save_path"] = save_path
        
        return result
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/base64")
async def predict_base64(
    background_tasks: BackgroundTasks,
    base64_image: str = Form(...),
    save_image: bool = Form(False)
) -> JSONResponse:
    """
    Predict defects in base64-encoded image.
    
    Args:
        base64_image (str): Base64-encoded image data
        save_image (bool): Whether to save the image to disk
        
    Returns:
        dict: Prediction results
    """
    try:
        # Decode base64 image
        if "base64," in base64_image:
            base64_image = base64_image.split("base64,")[1]
        
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
        
        # Save image if requested
        save_path = None
        if save_image:
            save_path = save_upload(image)
        
        # Run prediction
        result = predict(image)
        
        # Add metadata
        result["timestamp"] = datetime.now().isoformat()
        
        if save_path:
            result["save_path"] = save_path
        
        return result
    except Exception as e:
        logger.error(f"Error processing base64 image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/structures")
async def get_structures():
    """Get list of monitored structures."""
    return {
        "structures": SAMPLE_DATA["structures"]
    }

@app.get("/api/alerts")
async def get_alerts():
    """Get recent alerts."""
    return {
        "alerts": SAMPLE_DATA["alerts"]
    }

@app.get("/api/inspections")
async def get_inspections():
    """Get recent inspections."""
    return {
        "inspections": SAMPLE_DATA["recent_inspections"]
    }

@app.get("/api/statistics")
async def get_statistics():
    """Get monitoring statistics."""
    return {
        "total_structures": SAMPLE_DATA["total_structures"],
        "healthy_structures": SAMPLE_DATA["healthy_structures"],
        "warning_structures": SAMPLE_DATA["warning_structures"],
        "critical_structures": SAMPLE_DATA["critical_structures"]
    }

@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    return {
        "model_loaded": model is not None,
        "device": str(device),
        "num_classes": len(class_names),
        "class_names": class_names,
        "uptime": "12:34:56",  # Placeholder
        "processed_images": 1234,  # Placeholder
        "defects_detected": 56  # Placeholder
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/simulate_drone")
async def simulate_drone(
    file: UploadFile = File(...),
    perspective: float = Form(0.2),
    rotation: float = Form(0.0)
):
    """
    Simulate drone view of an image.
    
    Args:
        file (UploadFile): Image file
        perspective (float): Perspective distortion amount (0.0-1.0)
        rotation (float): Rotation angle in degrees
        
    Returns:
        dict: Base64-encoded simulated image
    """
    try:
        # Read image
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Simulate drone view
        simulated_img = simulate_drone_image(
            img,
            perspective_amount=perspective,
            rotation_angle=rotation
        )
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', simulated_img)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "base64_image": f"data:image/png;base64,{base64_image}",
            "parameters": {
                "perspective": perspective,
                "rotation": rotation
            }
        }
    except Exception as e:
        logger.error(f"Error simulating drone view: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/new_inspection")
async def create_inspection(
    structure_id: int = Form(...),
    inspection_type: str = Form(...),
    notes: str = Form(None),
    automatic_capture: bool = Form(True)
):
    """Create a new inspection."""
    # In a real app, this would create a new inspection in the database
    structure = next((s for s in SAMPLE_DATA["structures"] if s["id"] == structure_id), None)
    if not structure:
        raise HTTPException(status_code=404, detail="Structure not found")
    
    return {
        "id": f"insp-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4]}",
        "structure_name": structure["name"],
        "date": datetime.now().isoformat(),
        "status": "Pending",
        "inspection_type": inspection_type,
        "notes": notes,
        "automatic_capture": automatic_capture
    }

@app.get("/reports", response_class=HTMLResponse)
async def reports(request: Request):
    """Render the reports page."""
    try:
        # Get data from services for consistent UI
        statistics = await get_statistics_async()
        
        # Build template context
        data = {
            "request": request,
            "current_date": datetime.now().strftime("%B %d, %Y"),
            **statistics,
            **SAMPLE_DATA,
            "page_title": "Reports & Analytics",
            "active_page": "reports"
        }
        
        logger.info(f"Rendering reports page with data keys: {list(data.keys())}")
        # Use the simplified template that doesn't rely on Chart.js
        return templates.TemplateResponse("simple_reports.html", data)
    except Exception as e:
        logger.error(f"Error rendering reports page: {str(e)}", exc_info=True)
        return HTMLResponse(content=f"""
        <html>
            <head><title>Error</title></head>
            <body>
                <h1>Error rendering reports page</h1>
                <p>{str(e)}</p>
                <p>Check server logs for more details.</p>
                <p><a href="/">Return to Home</a></p>
            </body>
        </html>
        """)

@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    """Render the settings page."""
    # Get data from services for consistent UI
    statistics = get_statistics()
    
    # Build template context
    data = {
        "request": request,
        "current_date": datetime.now().strftime("%B %d, %Y"),
        **statistics,
        "page_title": "System Settings",
        "active_page": "settings"
    }
    
    return templates.TemplateResponse("settings.html", data)

@app.get("/analysis", response_class=HTMLResponse)
async def analysis(request: Request):
    """Render the analysis page."""
    # Get data from services for consistent UI
    statistics = get_statistics()
    
    # Build template context
    data = {
        "request": request,
        "current_date": datetime.now().strftime("%B %d, %Y"),
        **statistics,
        "page_title": "Structural Analysis",
        "active_page": "analysis"
    }
    
    return templates.TemplateResponse("analysis.html", data)

# Add a new async version of get_statistics
async def get_statistics_async():
    """Async version of get_statistics."""
    return {
        "total_structures": SAMPLE_DATA["total_structures"],
        "healthy_structures": SAMPLE_DATA["healthy_structures"],
        "warning_structures": SAMPLE_DATA["warning_structures"],
        "critical_structures": SAMPLE_DATA["critical_structures"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 