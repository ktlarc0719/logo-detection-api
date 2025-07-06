import asyncio
import time
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from src.core.config import get_settings
from src.core.detection_engine import get_detection_engine, DetectionEngineError
from src.models.schemas import (
    BatchProcessingRequest, BatchProcessingResponse, ImageInput, ImageResult, 
    ImageError, Detection, ProcessingOptions
)
from src.utils.image_utils import ImageProcessor, ImageDownloadError, ImageProcessingError
from src.utils.logger import get_logger
from src.utils.metrics import get_metrics_collector, MetricsTimer, BatchMetrics

logger = get_logger(__name__)


class BatchProcessorError(Exception):
    """Exception raised by batch processor."""
    pass


class AsyncBatchProcessor:
    """Async batch processor for logo detection."""
    
    def __init__(self):
        self.settings = get_settings()
        self.detection_engine = get_detection_engine()
        self.metrics_collector = get_metrics_collector()
        
        # Thread pool for CPU-bound detection tasks
        self.detection_executor = ThreadPoolExecutor(
            max_workers=self.settings.max_concurrent_detections,
            thread_name_prefix="detection"
        )
    
    async def process_batch(self, request: BatchProcessingRequest) -> BatchProcessingResponse:
        """
        Process a batch of images asynchronously.
        
        Args:
            request: Batch processing request
            
        Returns:
            Batch processing response with results and errors
        """
        logger.info(f"Starting batch processing: {request.batch_id} ({len(request.images)} images)")
        
        # Start metrics tracking
        batch_metrics = self.metrics_collector.start_batch(
            request.batch_id, 
            len(request.images)
        )
        
        try:
            with MetricsTimer(f"batch_processing_{request.batch_id}"):
                # Process images concurrently
                results, errors = await self._process_images_concurrently(
                    request.images,
                    request.options,
                    batch_metrics
                )
                
                # Finish metrics tracking
                completed_batch = self.metrics_collector.finish_batch(request.batch_id)
                processing_time = completed_batch.processing_time if completed_batch else 0.0
                
                # Create response
                response = BatchProcessingResponse(
                    batch_id=request.batch_id,
                    processing_time=processing_time,
                    total_images=len(request.images),
                    successful=len(results),
                    failed=len(errors),
                    results=results,
                    errors=errors
                )
                
                logger.info(
                    f"Batch processing completed: {request.batch_id} "
                    f"(successful: {len(results)}, failed: {len(errors)}, "
                    f"time: {processing_time:.2f}s)"
                )
                
                return response
                
        except Exception as e:
            logger.error(f"Batch processing failed: {request.batch_id} - {str(e)}")
            
            # Finish metrics tracking with error
            self.metrics_collector.finish_batch(request.batch_id)
            
            raise BatchProcessorError(f"Batch processing failed: {str(e)}")
    
    async def _process_images_concurrently(
        self,
        images: List[ImageInput],
        options: ProcessingOptions,
        batch_metrics: BatchMetrics
    ) -> Tuple[List[ImageResult], List[ImageError]]:
        """Process images with controlled concurrency."""
        
        # Create semaphore to limit concurrent downloads
        download_semaphore = asyncio.Semaphore(self.settings.max_concurrent_downloads)
        
        # Create tasks for all images
        tasks = [
            self._process_single_image(
                image,
                options,
                download_semaphore,
                batch_metrics
            )
            for image in images
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful results from errors
        successful_results = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error = ImageError(
                    id=images[i].id,
                    error=str(result)
                )
                errors.append(error)
                batch_metrics.add_error(images[i].id, str(result))
            elif isinstance(result, ImageResult):
                successful_results.append(result)
                batch_metrics.add_success()
            else:
                # Unexpected result type
                error = ImageError(
                    id=images[i].id,
                    error="Unexpected result type"
                )
                errors.append(error)
                batch_metrics.add_error(images[i].id, "Unexpected result type")
        
        return successful_results, errors
    
    async def _process_single_image(
        self,
        image_input: ImageInput,
        options: ProcessingOptions,
        download_semaphore: asyncio.Semaphore,
        batch_metrics: BatchMetrics
    ) -> ImageResult:
        """Process a single image through the complete pipeline."""
        start_time = time.time()
        
        try:
            # Download and preprocess image
            async with download_semaphore:
                image_array, image_info = await self._download_and_preprocess_image(
                    str(image_input.image_url)
                )
            
            # Run detection (CPU-bound, run in thread pool)
            detections = await self._run_detection(
                image_array,
                options.confidence_threshold,
                options.max_detections,
                options.model_name
            )
            
            processing_time = time.time() - start_time
            self.metrics_collector.record_processing_time(processing_time)
            
            return ImageResult(
                id=image_input.id,
                detections=detections,
                processing_time=processing_time,
                status="success"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_message = str(e)
            
            logger.warning(f"Failed to process image {image_input.id}: {error_message}")
            
            # Record error metrics
            error_type = type(e).__name__
            self.metrics_collector.record_error(error_type, error_message)
            
            return ImageResult(
                id=image_input.id,
                detections=[],
                processing_time=processing_time,
                status="failed",
                error_message=error_message
            )
    
    async def _download_and_preprocess_image(self, url: str) -> Tuple[np.ndarray, Dict]:
        """Download and preprocess image from URL."""
        try:
            async with ImageProcessor() as processor:
                image_array, image_info = await processor.process_image_url(url)
                return image_array, image_info
                
        except (ImageDownloadError, ImageProcessingError) as e:
            raise e
        except Exception as e:
            raise ImageProcessingError(f"Unexpected error processing image: {str(e)}")
    
    async def _run_detection(
        self,
        image: np.ndarray,
        confidence_threshold: float,
        max_detections: int,
        model_name: Optional[str] = None
    ) -> List[Detection]:
        """Run logo detection on preprocessed image."""
        if not self.detection_engine.is_loaded(model_name):
            model_to_check = model_name or self.detection_engine.current_model_name
            raise DetectionEngineError(f"Detection model '{model_to_check}' not loaded")
        
        try:
            # Run detection in thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            detections = await loop.run_in_executor(
                self.detection_executor,
                self.detection_engine.detect,
                image,
                confidence_threshold,
                max_detections,
                model_name
            )
            
            return detections
            
        except DetectionEngineError as e:
            raise e
        except Exception as e:
            raise DetectionEngineError(f"Detection failed: {str(e)}")
    
    async def process_single_image(
        self,
        image_url: str,
        confidence_threshold: float = 0.8,
        max_detections: int = 10
    ) -> ImageResult:
        """Process a single image (utility method for single image endpoint)."""
        
        # Create temporary image input
        image_input = ImageInput(
            id=0,
            image_url=image_url
        )
        
        # Create processing options
        options = ProcessingOptions(
            confidence_threshold=confidence_threshold,
            max_detections=max_detections
        )
        
        # Create temporary batch metrics
        batch_metrics = self.metrics_collector.start_batch("single_image", 1)
        
        try:
            # Create semaphore for single image
            download_semaphore = asyncio.Semaphore(1)
            
            # Process the image
            result = await self._process_single_image(
                image_input,
                options,
                download_semaphore,
                batch_metrics
            )
            
            # Finish metrics
            self.metrics_collector.finish_batch("single_image")
            
            return result
            
        except Exception as e:
            self.metrics_collector.finish_batch("single_image")
            raise e
    
    def get_status(self) -> Dict:
        """Get current processor status."""
        return {
            "detection_engine_loaded": self.detection_engine.is_loaded(),
            "max_concurrent_downloads": self.settings.max_concurrent_downloads,
            "max_concurrent_detections": self.settings.max_concurrent_detections,
            "max_batch_size": self.settings.max_batch_size,
            "active_threads": self.detection_executor._threads,
            "pending_tasks": self.detection_executor._work_queue.qsize() if hasattr(self.detection_executor._work_queue, 'qsize') else 0
        }
    
    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'detection_executor') and self.detection_executor:
                self.detection_executor.shutdown(wait=False)
        except Exception:
            pass


# Global batch processor instance
_batch_processor: Optional[AsyncBatchProcessor] = None


def get_batch_processor() -> AsyncBatchProcessor:
    """Get the global batch processor instance (singleton)."""
    global _batch_processor
    
    if _batch_processor is None:
        _batch_processor = AsyncBatchProcessor()
    
    return _batch_processor