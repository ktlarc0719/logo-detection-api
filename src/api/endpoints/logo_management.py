"""
Logo Management API Endpoints

This module provides REST API endpoints for logo and model management.
It includes endpoints for:
- Logo class management
- Custom logo upload and training
- Model deployment and versioning
- Logo search and similarity
"""

import os
import shutil
import hashlib
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import cv2
import numpy as np
from PIL import Image

from src.core.config import get_settings
from src.core.detection_engine import get_detection_engine
from src.utils.dataset_manager import get_dataset_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter()


# Request/Response Models
class LogoClass(BaseModel):
    """Logo class model."""
    name: str = Field(..., description="Logo class name")
    description: Optional[str] = Field(None, description="Class description")
    category: Optional[str] = Field(None, description="Logo category")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    created: str = Field(..., description="Creation timestamp")
    image_count: int = Field(0, description="Number of training images")
    is_active: bool = Field(True, description="Whether class is active")


class LogoUploadResponse(BaseModel):
    """Response model for logo upload."""
    success: bool
    message: str
    logo_id: str
    file_path: str
    processed: bool = False


class ModelInfo(BaseModel):
    """Model information model."""
    name: str
    path: str
    size_mb: float
    created: str
    version: str
    classes: List[str]
    accuracy_metrics: Optional[Dict[str, float]] = None
    is_deployed: bool = False


class LogoSearchRequest(BaseModel):
    """Request model for logo search."""
    query: str = Field(..., description="Search query")
    category: Optional[str] = Field(None, description="Filter by category")
    limit: int = Field(10, ge=1, le=100, description="Maximum results")


class LogoSimilarityRequest(BaseModel):
    """Request model for logo similarity search."""
    reference_logo: str = Field(..., description="Reference logo name or ID")
    threshold: float = Field(0.8, ge=0.0, le=1.0, description="Similarity threshold")
    limit: int = Field(10, ge=1, le=50, description="Maximum results")


# Logo Class Management
@router.get("/logos/classes")
async def list_logo_classes(
    category: Optional[str] = Query(None, description="Filter by category"),
    active_only: bool = Query(True, description="Show only active classes")
) -> List[LogoClass]:
    """List all logo classes."""
    try:
        settings = get_settings()
        dataset_manager = get_dataset_manager()
        
        # Get all datasets to find logo classes
        datasets = dataset_manager.list_datasets()
        
        all_classes = {}
        
        for dataset_info in datasets:
            if not dataset_info["is_valid"]:
                continue
            
            dataset_stats = dataset_manager.get_dataset_statistics(dataset_info["name"])
            if dataset_stats["success"]:
                for class_name in dataset_stats["class_names"]:
                    if class_name not in all_classes:
                        all_classes[class_name] = LogoClass(
                            name=class_name,
                            description=f"Logo class for {class_name}",
                            category="default",
                            aliases=[],
                            created=dataset_info["created"],
                            image_count=dataset_stats["classes"].get(class_name, 0),
                            is_active=True
                        )
                    else:
                        # Update image count
                        all_classes[class_name].image_count += dataset_stats["classes"].get(class_name, 0)
        
        # Apply filters
        result = list(all_classes.values())
        
        if category:
            result = [cls for cls in result if cls.category == category]
        
        if active_only:
            result = [cls for cls in result if cls.is_active]
        
        return sorted(result, key=lambda x: x.name)
        
    except Exception as e:
        logger.error(f"Failed to list logo classes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list logo classes: {str(e)}")


@router.post("/logos/classes")
async def create_logo_class(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    category: Optional[str] = Form("default"),
    aliases: Optional[str] = Form(None)  # Comma-separated
) -> Dict[str, Any]:
    """Create a new logo class."""
    try:
        dataset_manager = get_dataset_manager()
        
        # Parse aliases
        alias_list = []
        if aliases:
            alias_list = [alias.strip() for alias in aliases.split(",") if alias.strip()]
        
        # Create a dataset for this logo class
        dataset_name = f"logo_class_{name.lower().replace(' ', '_')}"
        
        # Check if class already exists
        existing_datasets = dataset_manager.list_datasets()
        for dataset in existing_datasets:
            if dataset["name"] == dataset_name:
                raise HTTPException(status_code=400, detail=f"Logo class '{name}' already exists")
        
        # Create dataset
        result = dataset_manager.create_dataset_structure(dataset_name)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Add the class to the dataset
        class_result = dataset_manager.add_class_to_dataset(dataset_name, name)
        if not class_result["success"]:
            dataset_manager.delete_dataset(dataset_name)
            raise HTTPException(status_code=400, detail=class_result["error"])
        
        logger.info(f"Created logo class '{name}' with dataset '{dataset_name}'")
        
        return {
            "success": True,
            "message": f"Logo class '{name}' created successfully",
            "class_name": name,
            "dataset_name": dataset_name,
            "description": description,
            "category": category,
            "aliases": alias_list
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create logo class: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create logo class: {str(e)}")


@router.delete("/logos/classes/{class_name}")
async def delete_logo_class(class_name: str) -> Dict[str, Any]:
    """Delete a logo class and its associated data."""
    try:
        dataset_manager = get_dataset_manager()
        
        # Find dataset containing this class
        datasets = dataset_manager.list_datasets()
        target_dataset = None
        
        for dataset_info in datasets:
            if dataset_info["is_valid"]:
                dataset_stats = dataset_manager.get_dataset_statistics(dataset_info["name"])
                if dataset_stats["success"] and class_name in dataset_stats["class_names"]:
                    target_dataset = dataset_info["name"]
                    break
        
        if not target_dataset:
            raise HTTPException(status_code=404, detail=f"Logo class '{class_name}' not found")
        
        # Delete the dataset
        result = dataset_manager.delete_dataset(target_dataset)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        logger.info(f"Deleted logo class '{class_name}' and dataset '{target_dataset}'")
        
        return {
            "success": True,
            "message": f"Logo class '{class_name}' deleted successfully",
            "deleted_dataset": target_dataset
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete logo class: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete logo class: {str(e)}")


# Logo Upload and Management
@router.post("/logos/upload")
async def upload_logo_image(
    class_name: str = Form(...),
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    bbox: Optional[str] = Form(None),  # JSON string: [x_min, y_min, x_max, y_max]
    auto_annotate: bool = Form(True)
) -> LogoUploadResponse:
    """Upload a logo image for training."""
    try:
        settings = get_settings()
        dataset_manager = get_dataset_manager()
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Find dataset containing this class
        datasets = dataset_manager.list_datasets()
        target_dataset = None
        
        for dataset_info in datasets:
            if dataset_info["is_valid"]:
                dataset_stats = dataset_manager.get_dataset_statistics(dataset_info["name"])
                if dataset_stats["success"] and class_name in dataset_stats["class_names"]:
                    target_dataset = dataset_info["name"]
                    break
        
        if not target_dataset:
            raise HTTPException(status_code=404, detail=f"Logo class '{class_name}' not found")
        
        # Create temp directory for upload
        temp_dir = Path("temp/uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_hash = hashlib.md5(f"{class_name}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        file_ext = Path(file.filename).suffix or ".jpg"
        temp_filename = f"{class_name}_{file_hash}{file_ext}"
        temp_file_path = temp_dir / temp_filename
        
        # Save uploaded file
        content = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(content)
        
        # Validate image
        try:
            with Image.open(temp_file_path) as img:
                img_width, img_height = img.size
                if img_width < 32 or img_height < 32:
                    raise HTTPException(status_code=400, detail="Image too small (minimum 32x32)")
        except Exception as e:
            temp_file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Process bounding box
        annotations = []
        
        if bbox:
            try:
                import json
                bbox_coords = json.loads(bbox)
                if len(bbox_coords) != 4:
                    raise ValueError("Bounding box must have 4 coordinates")
                
                # Validate bbox coordinates
                x_min, y_min, x_max, y_max = bbox_coords
                if not (0 <= x_min < x_max <= img_width and 0 <= y_min < y_max <= img_height):
                    raise ValueError("Invalid bounding box coordinates")
                
                annotations.append({
                    "class_name": class_name,
                    "bbox": bbox_coords
                })
                
            except Exception as e:
                temp_file_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail=f"Invalid bounding box: {str(e)}")
        
        elif auto_annotate:
            # Auto-generate bounding box (full image with some padding)
            padding = 10
            annotations.append({
                "class_name": class_name,
                "bbox": [padding, padding, img_width - padding, img_height - padding]
            })
        
        else:
            temp_file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Either bbox or auto_annotate must be provided")
        
        # Add image to dataset
        result = dataset_manager.add_image_with_annotation(
            dataset_name=target_dataset,
            image_path=str(temp_file_path),
            annotations=annotations,
            split="train"
        )
        
        # Clean up temp file
        temp_file_path.unlink(missing_ok=True)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        logger.info(f"Uploaded logo image for class '{class_name}' to dataset '{target_dataset}'")
        
        return LogoUploadResponse(
            success=True,
            message=f"Logo image uploaded successfully for class '{class_name}'",
            logo_id=file_hash,
            file_path=result["image_path"],
            processed=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logo upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Logo upload failed: {str(e)}")


@router.get("/logos/classes/{class_name}/images")
async def list_class_images(
    class_name: str,
    split: Optional[str] = Query(None, description="Filter by split (train/val/test)"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of images")
) -> List[Dict[str, Any]]:
    """List images for a specific logo class."""
    try:
        dataset_manager = get_dataset_manager()
        
        # Find dataset containing this class
        datasets = dataset_manager.list_datasets()
        target_dataset = None
        
        for dataset_info in datasets:
            if dataset_info["is_valid"]:
                dataset_stats = dataset_manager.get_dataset_statistics(dataset_info["name"])
                if dataset_stats["success"] and class_name in dataset_stats["class_names"]:
                    target_dataset = dataset_info["name"]
                    break
        
        if not target_dataset:
            raise HTTPException(status_code=404, detail=f"Logo class '{class_name}' not found")
        
        # Get dataset path
        dataset_path = Path(get_settings().training_data_dir) / target_dataset
        
        images = []
        splits_to_check = [split] if split else ["train", "val", "test"]
        
        for split_name in splits_to_check:
            images_dir = dataset_path / split_name / "images"
            labels_dir = dataset_path / split_name / "labels"
            
            if not images_dir.exists():
                continue
            
            for img_path in images_dir.iterdir():
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    label_path = labels_dir / f"{img_path.stem}.txt"
                    
                    # Check if this image contains the requested class
                    if label_path.exists():
                        with open(label_path, 'r') as f:
                            labels = f.read().strip().split('\n')
                        
                        # Get class ID for the requested class
                        dataset_yaml = dataset_path / "dataset.yaml"
                        if dataset_yaml.exists():
                            import yaml
                            with open(dataset_yaml, 'r') as f:
                                dataset_config = yaml.safe_load(f)
                            
                            if class_name in dataset_config["names"]:
                                class_id = dataset_config["names"].index(class_name)
                                
                                # Check if any label matches this class
                                for label in labels:
                                    if label and label.split()[0] == str(class_id):
                                        # Get image info
                                        stat = img_path.stat()
                                        
                                        images.append({
                                            "filename": img_path.name,
                                            "path": str(img_path),
                                            "split": split_name,
                                            "size_bytes": stat.st_size,
                                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                            "has_annotation": True
                                        })
                                        break
                    
                    if len(images) >= limit:
                        break
            
            if len(images) >= limit:
                break
        
        return images[:limit]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list class images: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list class images: {str(e)}")


@router.get("/logos/classes/{class_name}/images/{image_name}")
async def get_class_image(class_name: str, image_name: str) -> FileResponse:
    """Get a specific image file for a logo class."""
    try:
        dataset_manager = get_dataset_manager()
        
        # Find dataset and image
        datasets = dataset_manager.list_datasets()
        
        for dataset_info in datasets:
            if dataset_info["is_valid"]:
                dataset_stats = dataset_manager.get_dataset_statistics(dataset_info["name"])
                if dataset_stats["success"] and class_name in dataset_stats["class_names"]:
                    dataset_path = Path(get_settings().training_data_dir) / dataset_info["name"]
                    
                    # Search in all splits
                    for split in ["train", "val", "test"]:
                        image_path = dataset_path / split / "images" / image_name
                        if image_path.exists():
                            return FileResponse(
                                path=str(image_path),
                                media_type="image/jpeg",
                                filename=image_name
                            )
        
        raise HTTPException(status_code=404, detail="Image not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get class image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get class image: {str(e)}")


# Model Management
@router.get("/logos/models")
async def list_trained_models() -> List[ModelInfo]:
    """List all available trained models."""
    try:
        from src.core.training_engine import get_training_engine
        
        training_engine = get_training_engine()
        if not training_engine:
            return []
        
        models_data = training_engine._get_available_trained_models()
        
        result = []
        for model_data in models_data:
            model_info = ModelInfo(
                name=model_data["name"],
                path=model_data["path"],
                size_mb=model_data["size_mb"],
                created=model_data["created"],
                version=model_data.get("version", "1.0"),
                classes=model_data.get("classes", []),
                accuracy_metrics=model_data.get("accuracy_metrics"),
                is_deployed=model_data.get("is_deployed", False)
            )
            result.append(model_info)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to list trained models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list trained models: {str(e)}")


@router.post("/logos/models/{model_name}/deploy")
async def deploy_model(model_name: str) -> Dict[str, Any]:
    """Deploy a trained model as the active trademark model."""
    try:
        settings = get_settings()
        detection_engine = get_detection_engine()
        
        # Find the model file
        models_dir = Path(settings.training_output_dir)
        model_path = None
        
        for model_file in models_dir.glob("**/*.pt"):
            if model_file.stem == model_name or model_name in str(model_file):
                model_path = model_file
                break
        
        if not model_path or not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        # Copy to trademark model location
        trademark_model_path = Path("models/trademark_logos.pt")
        trademark_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(model_path, trademark_model_path)
        
        # Try to reload the trademark model in the detection engine
        try:
            if hasattr(detection_engine, 'switch_model'):
                result = detection_engine.switch_model("trademark")
                if not result:
                    logger.warning("Failed to switch to deployed model, but file was copied")
        except Exception as e:
            logger.warning(f"Model deployment copied file but failed to reload: {e}")
        
        logger.info(f"Deployed model '{model_name}' as trademark model")
        
        return {
            "success": True,
            "message": f"Model '{model_name}' deployed successfully",
            "deployed_path": str(trademark_model_path),
            "original_path": str(model_path)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model deployment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model deployment failed: {str(e)}")


# Logo Search and Similarity
@router.post("/logos/search")
async def search_logos(request: LogoSearchRequest) -> List[Dict[str, Any]]:
    """Search for logos by name or description."""
    try:
        classes = await list_logo_classes(category=request.category)
        
        results = []
        query_lower = request.query.lower()
        
        for logo_class in classes:
            score = 0
            
            # Exact name match
            if query_lower == logo_class.name.lower():
                score = 100
            # Name contains query
            elif query_lower in logo_class.name.lower():
                score = 80
            # Alias match
            elif any(query_lower in alias.lower() for alias in logo_class.aliases):
                score = 70
            # Description match
            elif logo_class.description and query_lower in logo_class.description.lower():
                score = 50
            
            if score > 0:
                results.append({
                    "logo_class": logo_class.dict(),
                    "match_score": score,
                    "match_reason": "name" if score >= 80 else "alias" if score == 70 else "description"
                })
        
        # Sort by score and limit results
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results[:request.limit]
        
    except Exception as e:
        logger.error(f"Logo search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Logo search failed: {str(e)}")


@router.get("/logos/statistics")
async def get_logo_statistics() -> Dict[str, Any]:
    """Get comprehensive logo management statistics."""
    try:
        dataset_manager = get_dataset_manager()
        
        # Get all datasets
        datasets = dataset_manager.list_datasets()
        
        total_classes = 0
        total_images = 0
        categories = {}
        class_distribution = {}
        
        for dataset_info in datasets:
            if dataset_info["is_valid"]:
                dataset_stats = dataset_manager.get_dataset_statistics(dataset_info["name"])
                if dataset_stats["success"]:
                    total_classes += dataset_stats["num_classes"]
                    total_images += dataset_stats["total_images"]
                    
                    for class_name, count in dataset_stats["classes"].items():
                        class_distribution[class_name] = class_distribution.get(class_name, 0) + count
        
        # Get model statistics
        try:
            models = await list_trained_models()
            total_models = len(models)
            deployed_models = len([m for m in models if m.is_deployed])
        except Exception:
            total_models = 0
            deployed_models = 0
        
        return {
            "total_classes": total_classes,
            "total_images": total_images,
            "total_datasets": len(datasets),
            "valid_datasets": len([d for d in datasets if d["is_valid"]]),
            "total_models": total_models,
            "deployed_models": deployed_models,
            "class_distribution": dict(sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)[:10]),
            "categories": categories,
            "average_images_per_class": total_images / total_classes if total_classes > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Failed to get logo statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get logo statistics: {str(e)}")