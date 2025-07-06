from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel, HttpUrl

from src.models.schemas import ImageResult
from src.core.batch_processor import get_batch_processor, BatchProcessorError
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class URLBatchRequest(BaseModel):
    """Request model for batch URL processing"""
    urls: List[HttpUrl]
    confidence_threshold: float = 0.5
    max_detections: int = 10


class URLBatchResponse(BaseModel):
    """Response model for batch URL processing"""
    total_urls: int
    processed: int
    failed: int
    results: Dict[str, ImageResult]
    processing_time: float


@router.post(
    "/urls/batch",
    response_model=URLBatchResponse,
    summary="Process batch of URLs for logo detection",
    description="Process multiple image URLs for logo detection. "
                "Returns detected logos for each URL with confidence scores and bounding boxes.",
    tags=["URL Batch Processing"]
)
async def process_url_batch(request: URLBatchRequest) -> URLBatchResponse:
    """
    Process a batch of image URLs for logo detection.
    
    - **urls**: List of image URLs to process
    - **confidence_threshold**: Minimum confidence score for detections (0.0-1.0)
    - **max_detections**: Maximum number of detections to return per image
    
    Returns detection results for each URL.
    """
    try:
        logger.info(f"Received URL batch processing request with {len(request.urls)} URLs")
        
        if len(request.urls) == 0:
            raise HTTPException(
                status_code=400,
                detail="URL list cannot be empty"
            )
        
        if len(request.urls) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 URLs allowed per batch"
            )
        
        # Get batch processor
        processor = get_batch_processor()
        
        # Process URLs concurrently
        import time
        start_time = time.time()
        
        results = {}
        processed = 0
        failed = 0
        
        # Process each URL
        for url in request.urls:
            try:
                result = await processor.process_single_image(
                    str(url),
                    request.confidence_threshold,
                    request.max_detections
                )
                results[str(url)] = result
                if result.status == "success":
                    processed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Failed to process URL {url}: {str(e)}")
                failed += 1
                results[str(url)] = ImageResult(
                    detections=[],
                    processing_time=0.0,
                    status="error",
                    error_message=str(e)
                )
        
        processing_time = time.time() - start_time
        
        response = URLBatchResponse(
            total_urls=len(request.urls),
            processed=processed,
            failed=failed,
            results=results,
            processing_time=processing_time
        )
        
        logger.info(f"URL batch processing completed: {processed} successful, {failed} failed")
        return response
        
    except BatchProcessorError as e:
        logger.error(f"Batch processor error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in URL batch processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during URL batch processing"
        )


@router.post(
    "/urls/from-file",
    response_model=URLBatchResponse,
    summary="Process URLs from a newline-delimited string",
    description="Process URLs provided as a newline-delimited string for logo detection.",
    tags=["URL Batch Processing"]
)
async def process_urls_from_text(
    urls_text: str,
    confidence_threshold: float = 0.5,
    max_detections: int = 10
) -> URLBatchResponse:
    """
    Process URLs from a newline-delimited text.
    
    - **urls_text**: Newline-delimited string of URLs
    - **confidence_threshold**: Minimum confidence score for detections (0.0-1.0)
    - **max_detections**: Maximum number of detections to return per image
    
    Returns detection results for each URL.
    """
    try:
        # Parse URLs from text
        urls = [url.strip() for url in urls_text.strip().split('\n') if url.strip()]
        
        # Validate URLs
        validated_urls = []
        for url in urls:
            try:
                # Basic URL validation
                if url.startswith(('http://', 'https://')):
                    validated_urls.append(url)
                else:
                    logger.warning(f"Skipping invalid URL: {url}")
            except Exception as e:
                logger.warning(f"Failed to validate URL {url}: {str(e)}")
        
        # Create request object
        request = URLBatchRequest(
            urls=validated_urls,
            confidence_threshold=confidence_threshold,
            max_detections=max_detections
        )
        
        # Process using the main batch function
        return await process_url_batch(request)
        
    except Exception as e:
        logger.error(f"Error processing URLs from text: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process URLs: {str(e)}"
        )