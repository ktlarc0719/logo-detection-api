from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any

from src.core.detection_engine import get_detection_engine, DetectionEngineError
from src.utils.brand_classifier import get_brand_classifier
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/models",
    response_model=Dict[str, Any],
    summary="Get available models information",
    description="Get information about all available detection models and their status.",
    tags=["Model Management"]
)
async def get_available_models() -> Dict[str, Any]:
    """
    Get information about all available detection models.
    
    Returns:
    - Model information including availability, status, and capabilities
    - Current active model
    - Configuration details
    """
    try:
        detection_engine = get_detection_engine()
        models_info = detection_engine.get_available_models()
        current_model_info = detection_engine.get_model_info()
        
        return {
            "current_model": detection_engine.current_model_name,
            "models": models_info,
            "current_model_info": current_model_info,
            "total_models": len(models_info),
            "loaded_models": sum(1 for model in models_info.values() if model.get("loaded", False))
        }
        
    except Exception as e:
        logger.error(f"Failed to get models information: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve models information"
        )


@router.get(
    "/models/current",
    response_model=Dict[str, Any],
    summary="Get current model information",
    description="Get detailed information about the currently active model.",
    tags=["Model Management"]
)
async def get_current_model() -> Dict[str, Any]:
    """
    Get detailed information about the currently active model.
    
    Returns detailed model information including capabilities and status.
    """
    try:
        detection_engine = get_detection_engine()
        model_info = detection_engine.get_model_info()
        
        return model_info
        
    except Exception as e:
        logger.error(f"Failed to get current model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve current model information"
        )


@router.post(
    "/models/switch",
    response_model=Dict[str, Any],
    summary="Switch to a different model",
    description="Switch the active detection model to a different one.",
    tags=["Model Management"]
)
async def switch_model(model_name: str) -> Dict[str, Any]:
    """
    Switch to a different detection model.
    
    Args:
    - **model_name**: Name of the model to switch to
    
    Returns status of the model switch operation.
    """
    try:
        detection_engine = get_detection_engine()
        
        # Validate model name
        available_models = detection_engine.get_available_models()
        if model_name not in available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' not available. Available models: {list(available_models.keys())}"
            )
        
        # Attempt to switch model
        success = detection_engine.switch_model(model_name)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to switch to model '{model_name}'"
            )
        
        # Get updated model info
        new_model_info = detection_engine.get_model_info()
        
        logger.info(f"Successfully switched to model: {model_name}")
        
        return {
            "success": True,
            "message": f"Successfully switched to model '{model_name}'",
            "previous_model": available_models.get(model_name, {}).get("is_current", False),
            "new_model": model_name,
            "model_info": new_model_info
        }
        
    except HTTPException:
        raise
    except DetectionEngineError as e:
        logger.error(f"Detection engine error switching model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection engine error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error switching model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during model switch"
        )


@router.post(
    "/models/{model_name}/load",
    response_model=Dict[str, Any],
    summary="Load a specific model",
    description="Load a specific model without switching to it.",
    tags=["Model Management"]
)
async def load_model(model_name: str) -> Dict[str, Any]:
    """
    Load a specific model without switching to it.
    
    Args:
    - **model_name**: Name of the model to load
    
    Returns status of the model loading operation.
    """
    try:
        detection_engine = get_detection_engine()
        
        # Validate model name
        available_models = detection_engine.get_available_models()
        if model_name not in available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' not available"
            )
        
        # Check if already loaded
        if detection_engine.is_loaded(model_name):
            return {
                "success": True,
                "message": f"Model '{model_name}' is already loaded",
                "model_name": model_name,
                "was_loaded": True
            }
        
        # Load the model
        detection_engine._load_model(model_name)
        
        logger.info(f"Successfully loaded model: {model_name}")
        
        return {
            "success": True,
            "message": f"Successfully loaded model '{model_name}'",
            "model_name": model_name,
            "was_loaded": False
        }
        
    except HTTPException:
        raise
    except DetectionEngineError as e:
        logger.error(f"Detection engine error loading model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection engine error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error loading model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during model loading"
        )


@router.get(
    "/brands",
    response_model=List[Dict[str, Any]],
    summary="Get available brands",
    description="Get list of all brands that can be detected and classified.",
    tags=["Brand Management"]
)
async def get_available_brands() -> List[Dict[str, Any]]:
    """
    Get list of all available brands for detection.
    
    Returns list of brands with their localized names and categories.
    """
    try:
        brand_classifier = get_brand_classifier()
        brands = brand_classifier.get_available_brands()
        
        return brands
        
    except Exception as e:
        logger.error(f"Failed to get available brands: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve available brands"
        )


@router.get(
    "/categories",
    response_model=List[Dict[str, Any]],
    summary="Get brand categories",
    description="Get list of all brand categories with subcategories and brand counts.",
    tags=["Brand Management"]
)
async def get_brand_categories() -> List[Dict[str, Any]]:
    """
    Get list of all brand categories.
    
    Returns hierarchical list of categories with subcategories and brand counts.
    """
    try:
        brand_classifier = get_brand_classifier()
        categories = brand_classifier.get_available_categories()
        
        return categories
        
    except Exception as e:
        logger.error(f"Failed to get brand categories: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve brand categories"
        )


@router.get(
    "/brands/{brand_name}/info",
    response_model=Dict[str, Any],
    summary="Get brand information",
    description="Get detailed information about a specific brand including localization and category.",
    tags=["Brand Management"]
)
async def get_brand_info(brand_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific brand.
    
    Args:
    - **brand_name**: Name of the brand to get information for
    
    Returns detailed brand information including normalized names and category.
    """
    try:
        brand_classifier = get_brand_classifier()
        
        # Normalize the brand name
        brand_info = brand_classifier.normalize_brand_name(brand_name)
        
        # Get category information
        category_info = brand_classifier.get_brand_category(brand_info["normalized"])
        
        return {
            "brand_info": brand_info,
            "category_info": category_info,
            "confidence_adjustment": brand_classifier.adjust_confidence_by_category(
                brand_info["normalized"], 0.8
            ) - 0.8  # Show the adjustment amount
        }
        
    except Exception as e:
        logger.error(f"Failed to get brand info for '{brand_name}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve information for brand '{brand_name}'"
        )


@router.post(
    "/brands/reload",
    response_model=Dict[str, str],
    summary="Reload brand classification data",
    description="Reload brand mapping and category data from files.",
    tags=["Brand Management"]
)
async def reload_brand_data() -> Dict[str, str]:
    """
    Reload brand classification data from files.
    
    Useful after updating brand mapping or category files.
    """
    try:
        brand_classifier = get_brand_classifier()
        brand_classifier.reload_data()
        
        logger.info("Brand classification data reloaded successfully")
        
        return {
            "status": "success",
            "message": "Brand classification data reloaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to reload brand data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to reload brand classification data"
        )