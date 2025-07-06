"""
Training API Endpoints

This module provides REST API endpoints for model training and dataset management.
It includes endpoints for:
- Training management (start, stop, status)
- Dataset management (create, validate, statistics)
- Model export and deployment
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

from src.core.training_engine import get_training_engine
from src.utils.dataset_manager import get_dataset_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter()


# Request/Response Models
class DatasetCreateRequest(BaseModel):
    """Request model for dataset creation."""
    name: str = Field(..., description="Dataset name")
    classes: List[str] = Field(..., description="List of class names")
    description: Optional[str] = Field(None, description="Dataset description")


class ImageAnnotation(BaseModel):
    """Image annotation model."""
    class_name: str = Field(..., description="Class name for the annotation")
    bbox: List[float] = Field(..., description="Bounding box [x_min, y_min, x_max, y_max]")
    confidence: Optional[float] = Field(1.0, description="Annotation confidence")


class AddImageRequest(BaseModel):
    """Request model for adding image to dataset."""
    image_path: str = Field(..., description="Path to image file")
    annotations: List[ImageAnnotation] = Field(..., description="List of annotations")
    split: str = Field("train", description="Dataset split (train/val/test)")
    preserve_filename: bool = Field(False, description="Preserve original filename instead of generating timestamp-based name")


class TrainingRequest(BaseModel):
    """Request model for starting training."""
    model_name: str = Field(..., description="Name for the trained model")
    dataset_name: str = Field(..., description="Dataset to use for training")
    base_model: str = Field("yolov8n.pt", description="Base model to start from")
    epochs: Optional[int] = Field(None, description="Number of training epochs")
    batch_size: Optional[int] = Field(None, description="Training batch size")
    learning_rate: Optional[float] = Field(None, description="Learning rate")


class DatasetSplitRequest(BaseModel):
    """Request model for dataset splitting."""
    train_ratio: float = Field(0.7, ge=0.1, le=0.9, description="Training set ratio")
    val_ratio: float = Field(0.2, ge=0.1, le=0.8, description="Validation set ratio")
    test_ratio: float = Field(0.1, ge=0.0, le=0.8, description="Test set ratio")


class ModelExportRequest(BaseModel):
    """Request model for model export."""
    model_path: str = Field(..., description="Path to model file")
    format: str = Field("onnx", description="Export format (onnx/tensorrt)")


# Dataset Management Endpoints
@router.post("/datasets/create")
async def create_dataset(request: DatasetCreateRequest) -> Dict[str, Any]:
    """Create a new dataset with specified classes."""
    try:
        dataset_manager = get_dataset_manager()
        
        # Create dataset structure
        result = dataset_manager.create_dataset_structure(request.name)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Add classes to dataset
        for class_name in request.classes:
            class_result = dataset_manager.add_class_to_dataset(request.name, class_name)
            if not class_result["success"]:
                # Clean up if class addition fails
                dataset_manager.delete_dataset(request.name)
                raise HTTPException(status_code=400, detail=class_result["error"])
        
        logger.info(f"Dataset '{request.name}' created with {len(request.classes)} classes")
        
        return {
            "success": True,
            "message": f"Dataset '{request.name}' created successfully",
            "dataset_name": request.name,
            "dataset_path": result["dataset_path"],
            "classes": request.classes,
            "total_classes": len(request.classes)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset creation failed: {str(e)}")


@router.get("/datasets")
async def list_datasets() -> List[Dict[str, Any]]:
    """List all available datasets."""
    try:
        dataset_manager = get_dataset_manager()
        datasets = dataset_manager.list_datasets()
        
        return datasets
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")


@router.get("/datasets/{dataset_name}/stats")
async def get_dataset_statistics(dataset_name: str) -> Dict[str, Any]:
    """Get detailed statistics for a dataset."""
    try:
        dataset_manager = get_dataset_manager()
        stats = dataset_manager.get_dataset_statistics(dataset_name)
        
        if not stats["success"]:
            raise HTTPException(status_code=404, detail=stats["error"])
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dataset statistics: {str(e)}")


@router.get("/datasets/{dataset_name}/validate")
async def validate_dataset(dataset_name: str) -> Dict[str, Any]:
    """Validate a dataset for training."""
    try:
        dataset_manager = get_dataset_manager()
        validation = dataset_manager.validate_dataset(dataset_name)
        
        if "success" in validation and not validation["success"]:
            raise HTTPException(status_code=404, detail=validation["error"])
        
        return validation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset validation failed: {str(e)}")


@router.post("/datasets/{dataset_name}/add-image")
async def add_image_to_dataset(dataset_name: str, request: AddImageRequest) -> Dict[str, Any]:
    """Add an image with annotations to a dataset."""
    try:
        dataset_manager = get_dataset_manager()
        
        # Convert annotations to expected format
        annotations = []
        for ann in request.annotations:
            annotations.append({
                "class_name": ann.class_name,
                "bbox": ann.bbox
            })
        
        result = dataset_manager.add_image_with_annotation(
            dataset_name=dataset_name,
            image_path=request.image_path,
            annotations=annotations,
            split=request.split,
            preserve_filename=request.preserve_filename
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add image to dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add image to dataset: {str(e)}")


@router.post("/datasets/{dataset_name}/split")
async def split_dataset(dataset_name: str, request: DatasetSplitRequest) -> Dict[str, Any]:
    """Split dataset into train/val/test sets."""
    try:
        dataset_manager = get_dataset_manager()
        
        result = dataset_manager.split_dataset(
            dataset_name=dataset_name,
            train_ratio=request.train_ratio,
            val_ratio=request.val_ratio,
            test_ratio=request.test_ratio
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset splitting failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset splitting failed: {str(e)}")


@router.delete("/datasets/{dataset_name}")
async def delete_dataset(dataset_name: str) -> Dict[str, Any]:
    """Delete a dataset."""
    try:
        dataset_manager = get_dataset_manager()
        result = dataset_manager.delete_dataset(dataset_name)
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset deletion failed: {str(e)}")


# Training Management Endpoints
@router.get("/training/status")
async def get_training_status() -> Dict[str, Any]:
    """Get current training status and progress."""
    try:
        training_engine = get_training_engine()
        
        if training_engine is None:
            return {
                "training_enabled": False,
                "message": "Training pipeline is disabled"
            }
        
        status = training_engine.get_training_status()
        status["training_enabled"] = True
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")


@router.post("/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Start model training."""
    try:
        training_engine = get_training_engine()
        
        if training_engine is None:
            raise HTTPException(status_code=503, detail="Training pipeline is disabled")
        
        # Validate dataset exists and is ready
        dataset_manager = get_dataset_manager()
        validation = dataset_manager.validate_dataset(request.dataset_name)
        
        if not validation["is_valid"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Dataset validation failed: {validation['errors']}"
            )
        
        # Start training (no need to prepare dataset, it already exists)
        result = await training_engine.start_training(
            model_name=request.model_name,
            dataset_name=request.dataset_name,
            base_model=request.base_model,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training start failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training start failed: {str(e)}")


@router.post("/training/stop")
async def stop_training() -> Dict[str, Any]:
    """Stop current training."""
    try:
        training_engine = get_training_engine()
        
        if training_engine is None:
            raise HTTPException(status_code=503, detail="Training pipeline is disabled")
        
        result = await training_engine.stop_training()
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training stop failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training stop failed: {str(e)}")


@router.get("/training/models")
async def list_trained_models() -> List[Dict[str, Any]]:
    """List available trained models."""
    try:
        training_engine = get_training_engine()
        
        if training_engine is None:
            return []
        
        models = training_engine._get_available_trained_models()
        return models
        
    except Exception as e:
        logger.error(f"Failed to list trained models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list trained models: {str(e)}")


@router.post("/training/export")
async def export_model(request: ModelExportRequest) -> Dict[str, Any]:
    """Export trained model to different formats."""
    try:
        training_engine = get_training_engine()
        
        if training_engine is None:
            raise HTTPException(status_code=503, detail="Training pipeline is disabled")
        
        result = training_engine.export_model(
            model_path=request.model_path,
            format=request.format
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model export failed: {str(e)}")


# Dataset Generation Endpoints
@router.post("/datasets/{dataset_name}/generate-sample")
async def generate_sample_dataset(
    dataset_name: str,
    classes: List[str] = Query(..., description="List of class names"),
    images_per_class: int = Query(20, ge=5, le=100, description="Number of images per class")
) -> Dict[str, Any]:
    """Generate a sample dataset with real logo images."""
    try:
        # Import real logo adapter instead of text-based generator
        from src.utils.real_logo_adapter import get_real_logo_adapter
        
        generator = get_real_logo_adapter()
        
        # Generate dataset using real logo images
        result = generator.generate_robust_dataset(
            dataset_name=dataset_name,
            class_names=classes,
            images_per_class=images_per_class
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sample dataset generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sample dataset generation failed: {str(e)}")


@router.post("/datasets/{dataset_name}/generate-from-logo")
async def generate_dataset_from_logo(
    dataset_name: str,
    request: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a dataset from a specific logo file path."""
    try:
        # Extract parameters
        logo_path = request.get("logo_path")
        brand_name = request.get("brand_name")
        num_variations = request.get("num_variations", 50)
        
        if not logo_path or not brand_name:
            raise HTTPException(
                status_code=400, 
                detail="logo_path and brand_name are required"
            )
        
        # Check if logo file exists
        from pathlib import Path
        if not Path(logo_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Logo file not found: {logo_path}"
            )
        
        # Use the real logo dataset generator directly
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent.parent))
        from real_logo_dataset_generator import RealLogoDatasetGenerator
        
        generator = RealLogoDatasetGenerator()
        
        # Generate dataset
        success = generator.create_dataset_with_variations(
            dataset_name=dataset_name,
            logo_path=logo_path,
            brand_name=brand_name,
            num_variations=num_variations
        )
        
        if success:
            return {
                "success": True,
                "message": f"Dataset '{dataset_name}' created successfully",
                "dataset_name": dataset_name,
                "brand_name": brand_name,
                "logo_path": logo_path,
                "num_variations": num_variations
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate dataset"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset generation from logo failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset generation failed: {str(e)}")


# Progress Monitoring Endpoints
@router.get("/training/progress")
async def get_training_progress() -> Dict[str, Any]:
    """Get detailed training progress information."""
    try:
        training_engine = get_training_engine()
        
        if training_engine is None:
            raise HTTPException(status_code=503, detail="Training pipeline is disabled")
        
        return training_engine.progress.to_dict()
        
    except Exception as e:
        logger.error(f"Failed to get training progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training progress: {str(e)}")


# WebSocket endpoint for real-time progress updates would go here
# This requires additional WebSocket setup in the main app