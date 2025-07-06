from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict

from src.models.schemas import (
    BatchProcessingRequest, BatchProcessingResponse, ErrorResponse
)
from src.core.batch_processor import get_batch_processor, BatchProcessorError
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/batch",
    response_model=BatchProcessingResponse,
    summary="Process batch of images for logo detection",
    description="Process multiple images concurrently for logo detection. "
                "Returns results for all images including successful detections and errors.",
    tags=["Batch Processing"]
)
async def process_batch(request: BatchProcessingRequest) -> BatchProcessingResponse:
    """
    Process a batch of images for logo detection.
    
    - **batch_id**: Unique identifier for this batch
    - **images**: List of images to process (max 100 per batch)
    - **options**: Processing options including confidence threshold and max detections
    
    Returns detailed results for each image including detections, processing times, and errors.
    """
    try:
        logger.info(f"Received batch processing request: {request.batch_id}")
        
        # Validate batch size
        if len(request.images) == 0:
            raise HTTPException(
                status_code=400,
                detail="Batch must contain at least one image"
            )
        
        # Get batch processor and process the batch
        processor = get_batch_processor()
        response = await processor.process_batch(request)
        
        logger.info(f"Batch processing completed: {request.batch_id}")
        return response
        
    except BatchProcessorError as e:
        logger.error(f"Batch processor error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in batch processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during batch processing"
        )


@router.get(
    "/batch/{batch_id}/status",
    response_model=Dict,
    summary="Get batch processing status",
    description="Get the current status of a batch processing operation.",
    tags=["Batch Processing"]
)
async def get_batch_status(batch_id: str) -> Dict:
    """
    Get the status of a batch processing operation.
    
    - **batch_id**: The unique identifier of the batch to check
    
    Returns current status including progress and any errors.
    """
    try:
        processor = get_batch_processor()
        status = processor.get_status()
        
        # Add batch-specific information if available
        # (This would be extended if we implement persistent batch tracking)
        return {
            "batch_id": batch_id,
            "processor_status": status,
            "message": "Batch status retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error retrieving batch status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve batch status"
        )