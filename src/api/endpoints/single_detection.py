from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import Dict, Any, List
from PIL import Image
import io

from src.models.schemas import (
    SingleImageRequest, SingleImageResponse, Detection
)
from src.core.batch_processor import get_batch_processor, BatchProcessorError
from src.utils.logger import get_logger
from src.utils.image_utils import get_image_info, ImageProcessingError

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/single",
    response_model=SingleImageResponse,
    summary="Process single image for logo detection",
    description="Process a single image for logo detection. "
                "Returns detected logos with confidence scores and bounding boxes.",
    tags=["Single Image Processing"]
)
async def process_single_image(request: SingleImageRequest) -> SingleImageResponse:
    """
    Process a single image for logo detection.
    
    - **image_url**: URL of the image to process
    - **confidence_threshold**: Minimum confidence score for detections (0.0-1.0)
    - **max_detections**: Maximum number of detections to return
    
    Returns detected logos with their confidence scores and bounding boxes.
    """
    try:
        logger.info(f"Received single image processing request: {request.image_url}")
        
        # Get batch processor and process single image
        processor = get_batch_processor()
        result = await processor.process_single_image(
            str(request.image_url),
            request.confidence_threshold,
            request.max_detections
        )
        
        # Convert ImageResult to SingleImageResponse
        response = SingleImageResponse(
            detections=result.detections,
            processing_time=result.processing_time,
            status=result.status,
            error_message=result.error_message
        )
        
        logger.info(f"Single image processing completed: {len(result.detections)} detections found")
        return response
        
    except BatchProcessorError as e:
        logger.error(f"Batch processor error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Image processing failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in single image processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during image processing"
        )


@router.post(
    "/single/info",
    response_model=Dict[str, Any],
    summary="Get image information without processing",
    description="Get basic information about an image without running detection.",
    tags=["Single Image Processing"]
)
async def get_image_information(request: SingleImageRequest) -> Dict[str, Any]:
    """
    Get information about an image without running detection.
    
    - **image_url**: URL of the image to analyze
    
    Returns image metadata including format, dimensions, and other properties.
    """
    try:
        logger.info(f"Received image info request: {request.image_url}")
        
        # Download and get image info
        from src.utils.image_utils import ImageProcessor
        
        async with ImageProcessor() as processor:
            try:
                # Download image
                image_data = await processor.session.get(str(request.image_url))
                image_bytes = await image_data.read()
                
                # Get image info
                info = get_image_info(image_bytes)
                
                logger.info(f"Image info retrieved successfully")
                return {
                    "status": "success",
                    "image_info": info,
                    "url": str(request.image_url)
                }
                
            except Exception as e:
                raise ImageProcessingError(f"Failed to get image info: {str(e)}")
        
    except ImageProcessingError as e:
        logger.error(f"Image processing error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error getting image info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting image information"
        )