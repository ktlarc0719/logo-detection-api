from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Logo Detection API")

# Global model variable
model = None

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

class DetectionResponse(BaseModel):
    detections: List[Detection]
    image_size: List[int]  # [width, height]

def load_model():
    """Load YOLO model"""
    global model
    try:
        # Use YOLOv8 nano model for faster inference
        model = YOLO('yolov8n.pt')
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

# Handle favicon requests
@app.get("/favicon.ico")
async def favicon():
    return Response(content="", status_code=204)

# Serve the main page
@app.get("/")
async def main():
    return FileResponse('static/index.html')

@app.post("/detect", response_model=DetectionResponse)
async def detect_logos(file: UploadFile = File(...)):
    """Detect logos in uploaded image"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get image size
        width, height = image.size
        
        # Run detection
        results = model(image)
        
        # Process results
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Get class name
                    class_name = model.names[cls]
                    
                    detection = Detection(
                        class_id=cls,
                        class_name=class_name,
                        confidence=conf,
                        bbox=[x1, y1, x2, y2]
                    )
                    detections.append(detection)
        
        return DetectionResponse(
            detections=detections,
            image_size=[width, height]
        )
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# Mount static files last to avoid conflicts
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)