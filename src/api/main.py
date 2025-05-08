from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import cv2
import numpy as np
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ai.models.defect_detector import DefectDetector

app = FastAPI(
    title="Structural Health Monitoring API",
    description="API for analyzing infrastructure defects using AI and drone imagery",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model
model = None

@app.on_event("startup")
async def load_model():
    """Load the AI model on startup."""
    global model
    try:
        model = DefectDetector(num_classes=4)
        # In production, load pre-trained weights here
        # model.load_state_dict(torch.load("path_to_weights.pth"))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "Structural Health Monitoring API",
        "status": "active",
        "version": "1.0.0"
    }

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an infrastructure image for defects.
    
    Args:
        file: Uploaded image file
    
    Returns:
        JSON response with defect analysis results
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        # Read and preprocess the image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Preprocess image for model
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image = image.unsqueeze(0) / 255.0
        
        # Make prediction
        with torch.no_grad():
            predictions, defect_map = model.predict(image)
        
        # Convert predictions to list
        defect_probs = predictions[0].tolist()
        defect_map = defect_map[0].squeeze().numpy().tolist()
        
        # Define defect classes
        defect_classes = ["No Defect", "Crack", "Corrosion", "Structural Damage"]
        
        # Prepare response
        results = {
            "defects": [
                {
                    "class": defect_classes[i],
                    "probability": defect_probs[i]
                } for i in range(len(defect_classes))
            ],
            "defect_map": defect_map
        }
        
        return JSONResponse(content=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 