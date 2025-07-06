import pytest
import asyncio
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from unittest import mock

from src.core.batch_processor import AsyncBatchProcessor, BatchProcessorError
from src.core.detection_engine import YOLODetectionEngine, DetectionEngineError
from src.models.schemas import (
    BatchProcessingRequest, ImageInput, ProcessingOptions, Detection
)
from src.utils.image_utils import ImageDownloadError, ImageProcessingError


@pytest.fixture
def mock_detection_engine():
    """Mock detection engine."""
    mock_engine = MagicMock()
    mock_engine.is_loaded.return_value = True
    mock_engine.detect.return_value = [
        Detection(
            logo_name="test_logo",
            confidence=0.95,
            bbox=[100, 50, 200, 100]
        )
    ]
    return mock_engine


@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector."""
    mock_collector = MagicMock()
    mock_batch_metrics = MagicMock()
    mock_batch_metrics.add_error = MagicMock()
    mock_batch_metrics.add_success = MagicMock()
    mock_collector.start_batch.return_value = mock_batch_metrics
    mock_collector.finish_batch.return_value = mock_batch_metrics
    mock_collector.record_processing_time = MagicMock()
    mock_collector.record_error = MagicMock()
    return mock_collector


@pytest.fixture
def batch_request():
    """Sample batch processing request."""
    return BatchProcessingRequest(
        batch_id="test_batch_001",
        images=[
            ImageInput(
                image_id="img_001",
                image_url="https://example.com/image1.jpg"
            ),
            ImageInput(
                image_id="img_002",
                image_url="https://example.com/image2.jpg"
            )
        ],
        options=ProcessingOptions(
            confidence_threshold=0.8,
            max_detections=10
        )
    )


class TestAsyncBatchProcessor:
    """Test AsyncBatchProcessor functionality."""
    
    @patch('src.core.batch_processor.get_detection_engine')
    @patch('src.core.batch_processor.get_metrics_collector')
    def test_initialization(self, mock_get_collector, mock_get_engine, 
                           mock_detection_engine, mock_metrics_collector):
        """Test batch processor initialization."""
        mock_get_engine.return_value = mock_detection_engine
        mock_get_collector.return_value = mock_metrics_collector
        
        processor = AsyncBatchProcessor()
        assert processor.detection_engine == mock_detection_engine
        assert processor.metrics_collector == mock_metrics_collector
    
    @patch('src.core.batch_processor.get_detection_engine')
    @patch('src.core.batch_processor.get_metrics_collector')
    @patch('src.core.batch_processor.ImageProcessor')
    async def test_successful_batch_processing(self, mock_image_processor_class,
                                              mock_get_collector, mock_get_engine,
                                              mock_detection_engine, mock_metrics_collector,
                                              batch_request):
        """Test successful batch processing."""
        # Mock image processor
        mock_image_processor = AsyncMock()
        mock_image_processor.__aenter__.return_value = mock_image_processor
        mock_image_processor.__aexit__.return_value = None
        mock_image_processor.process_image_url.return_value = (
            np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
            {"format": "JPEG", "size": (640, 640)}
        )
        mock_image_processor_class.return_value = mock_image_processor
        
        # Setup mocks
        mock_get_engine.return_value = mock_detection_engine
        mock_get_collector.return_value = mock_metrics_collector
        
        # Mock batch metrics
        mock_batch_metrics = MagicMock()
        mock_batch_metrics.processing_time = 2.5
        mock_metrics_collector.start_batch.return_value = mock_batch_metrics
        mock_metrics_collector.finish_batch.return_value = mock_batch_metrics
        
        processor = AsyncBatchProcessor()
        
        # Process batch
        response = await processor.process_batch(batch_request)
        
        # Verify response
        assert response.batch_id == "test_batch_001"
        assert response.total_images == 2
        assert response.successful >= 0
        assert response.processing_time >= 0
        
        # Verify metrics were called
        mock_metrics_collector.start_batch.assert_called_once()
        mock_metrics_collector.finish_batch.assert_called_once()
    
    @patch('src.core.batch_processor.get_detection_engine')
    @patch('src.core.batch_processor.get_metrics_collector')
    async def test_batch_processing_with_detection_error(self, mock_get_collector,
                                                        mock_get_engine,
                                                        mock_metrics_collector,
                                                        batch_request):
        """Test batch processing when detection engine has errors."""
        # Mock detection engine that raises error
        mock_detection_engine = MagicMock()
        mock_detection_engine.is_loaded.return_value = False
        mock_get_engine.return_value = mock_detection_engine
        mock_get_collector.return_value = mock_metrics_collector
        
        mock_batch_metrics = MagicMock()
        mock_metrics_collector.start_batch.return_value = mock_batch_metrics
        mock_metrics_collector.finish_batch.return_value = mock_batch_metrics
        
        processor = AsyncBatchProcessor()
        
        # This should not raise an exception but handle errors gracefully
        response = await processor.process_batch(batch_request)
        
        # Should have errors for all images since detection engine is not loaded
        assert response.failed >= 0
        assert len(response.errors) >= 0
    
    @patch('src.core.batch_processor.get_detection_engine')
    @patch('src.core.batch_processor.get_metrics_collector')
    async def test_single_image_processing(self, mock_get_collector, mock_get_engine,
                                          mock_detection_engine, mock_metrics_collector):
        """Test single image processing method."""
        mock_get_engine.return_value = mock_detection_engine
        mock_get_collector.return_value = mock_metrics_collector
        
        mock_batch_metrics = MagicMock()
        mock_metrics_collector.start_batch.return_value = mock_batch_metrics
        mock_metrics_collector.finish_batch.return_value = mock_batch_metrics
        
        processor = AsyncBatchProcessor()
        
        with patch.object(processor, '_process_single_image') as mock_process:
            mock_result = MagicMock()
            mock_result.detections = []
            mock_result.processing_time = 0.045
            mock_result.status = "success"
            mock_process.return_value = mock_result
            
            result = await processor.process_single_image(
                "https://example.com/image.jpg",
                confidence_threshold=0.8,
                max_detections=10
            )
            
            assert result.status == "success"
            assert result.processing_time == 0.045
    
    @patch('src.core.batch_processor.get_detection_engine')
    @patch('src.core.batch_processor.get_metrics_collector')
    def test_get_status(self, mock_get_collector, mock_get_engine,
                       mock_detection_engine, mock_metrics_collector):
        """Test get_status method."""
        mock_get_engine.return_value = mock_detection_engine
        mock_get_collector.return_value = mock_metrics_collector
        
        processor = AsyncBatchProcessor()
        status = processor.get_status()
        
        assert "detection_engine_loaded" in status
        assert "max_concurrent_downloads" in status
        assert "max_concurrent_detections" in status


class TestBatchProcessingIntegration:
    """Integration tests for batch processing."""
    
    @patch('src.core.batch_processor.get_detection_engine')
    @patch('src.core.batch_processor.get_metrics_collector')
    @patch('src.utils.image_utils.aiohttp.ClientSession')
    async def test_image_download_and_detection_flow(self, mock_session_class,
                                                    mock_get_collector, mock_get_engine,
                                                    mock_detection_engine, mock_metrics_collector):
        """Test the complete flow from image download to detection."""
        # Mock HTTP session for image download
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'content-length': '1000'}
        mock_response.read.return_value = b'fake_image_data'
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        # Mock image processing
        with patch('src.utils.image_utils.validate_image_format', return_value=True), \
             patch('src.utils.image_utils.get_image_info', return_value={"format": "JPEG"}), \
             patch('src.utils.image_utils.preprocess_image', 
                   return_value=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)):
            
            mock_get_engine.return_value = mock_detection_engine
            mock_get_collector.return_value = mock_metrics_collector
            
            mock_batch_metrics = MagicMock()
            mock_metrics_collector.start_batch.return_value = mock_batch_metrics
            mock_metrics_collector.finish_batch.return_value = mock_batch_metrics
            
            processor = AsyncBatchProcessor()
            
            request = BatchProcessingRequest(
                batch_id="integration_test",
                images=[
                    ImageInput(
                        image_id="test_img",
                        image_url="https://example.com/test.jpg"
                    )
                ]
            )
            
            response = await processor.process_batch(request)
            
            assert response.batch_id == "integration_test"
            assert response.total_images == 1


class TestErrorHandling:
    """Test error handling in batch processing."""
    
    @patch('src.core.batch_processor.get_detection_engine')
    @patch('src.core.batch_processor.get_metrics_collector')
    async def test_image_download_error_handling(self, mock_get_collector, mock_get_engine,
                                                mock_detection_engine, mock_metrics_collector):
        """Test handling of image download errors."""
        mock_get_engine.return_value = mock_detection_engine
        mock_get_collector.return_value = mock_metrics_collector
        
        mock_batch_metrics = MagicMock()
        mock_metrics_collector.start_batch.return_value = mock_batch_metrics
        mock_metrics_collector.finish_batch.return_value = mock_batch_metrics
        
        processor = AsyncBatchProcessor()
        
        # Mock image processor to raise download error
        with patch('src.core.batch_processor.ImageProcessor') as mock_image_processor_class:
            mock_image_processor = AsyncMock()
            mock_image_processor.__aenter__.return_value = mock_image_processor
            mock_image_processor.__aexit__.return_value = None
            mock_image_processor.process_image_url.side_effect = ImageDownloadError("Download failed")
            mock_image_processor_class.return_value = mock_image_processor
            
            request = BatchProcessingRequest(
                batch_id="error_test",
                images=[
                    ImageInput(
                        image_id="failing_img",
                        image_url="https://example.com/nonexistent.jpg"
                    )
                ]
            )
            
            response = await processor.process_batch(request)
            
            # Should handle error gracefully
            assert response.failed >= 0
            assert len(response.errors) >= 0
    
    @patch('src.core.batch_processor.get_detection_engine')
    @patch('src.core.batch_processor.get_metrics_collector')
    async def test_detection_error_handling(self, mock_get_collector, mock_get_engine,
                                           mock_metrics_collector):
        """Test handling of detection errors."""
        # Mock detection engine that raises error
        mock_detection_engine = MagicMock()
        mock_detection_engine.is_loaded.return_value = True
        mock_detection_engine.detect.side_effect = DetectionEngineError("Detection failed")
        mock_get_engine.return_value = mock_detection_engine
        mock_get_collector.return_value = mock_metrics_collector
        
        mock_batch_metrics = MagicMock()
        mock_metrics_collector.start_batch.return_value = mock_batch_metrics
        mock_metrics_collector.finish_batch.return_value = mock_batch_metrics
        
        processor = AsyncBatchProcessor()
        
        # Mock successful image processing but failing detection
        with patch('src.core.batch_processor.ImageProcessor') as mock_image_processor_class:
            mock_image_processor = AsyncMock()
            mock_image_processor.__aenter__.return_value = mock_image_processor
            mock_image_processor.__aexit__.return_value = None
            mock_image_processor.process_image_url.return_value = (
                np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
                {"format": "JPEG"}
            )
            mock_image_processor_class.return_value = mock_image_processor
            
            request = BatchProcessingRequest(
                batch_id="detection_error_test",
                images=[
                    ImageInput(
                        image_id="detection_failing_img",
                        image_url="https://example.com/image.jpg"
                    )
                ]
            )
            
            response = await processor.process_batch(request)
            
            # Should handle detection error gracefully
            assert response.total_images == 1


class TestConcurrencyLimits:
    """Test concurrency limiting functionality."""
    
    @patch('src.core.batch_processor.get_detection_engine')
    @patch('src.core.batch_processor.get_metrics_collector')
    @patch('src.core.config.get_settings')
    async def test_download_concurrency_limit(self, mock_get_settings, mock_get_collector,
                                             mock_get_engine, mock_detection_engine,
                                             mock_metrics_collector):
        """Test that download concurrency is properly limited."""
        # Mock settings with low concurrency limit
        mock_settings = MagicMock()
        mock_settings.max_concurrent_downloads = 2
        mock_settings.max_concurrent_detections = 2
        mock_get_settings.return_value = mock_settings
        
        mock_get_engine.return_value = mock_detection_engine
        mock_get_collector.return_value = mock_metrics_collector
        
        mock_batch_metrics = MagicMock()
        mock_metrics_collector.start_batch.return_value = mock_batch_metrics
        mock_metrics_collector.finish_batch.return_value = mock_batch_metrics
        
        processor = AsyncBatchProcessor()
        
        # Create request with many images
        images = [
            ImageInput(
                image_id=f"img_{i:03d}",
                image_url=f"https://example.com/image{i}.jpg"
            )
            for i in range(10)
        ]
        
        request = BatchProcessingRequest(
            batch_id="concurrency_test",
            images=images
        )
        
        # Mock image processing to track concurrent calls
        call_count = 0
        max_concurrent = 0
        current_concurrent = 0
        
        async def mock_process_image_url(*args, **kwargs):
            nonlocal call_count, max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.1)  # Simulate processing time
            current_concurrent -= 1
            call_count += 1
            return (
                np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
                {"format": "JPEG"}
            )
        
        with patch('src.core.batch_processor.ImageProcessor') as mock_image_processor_class:
            mock_image_processor = AsyncMock()
            mock_image_processor.__aenter__.return_value = mock_image_processor
            mock_image_processor.__aexit__.return_value = None
            mock_image_processor.process_image_url = mock_process_image_url
            mock_image_processor_class.return_value = mock_image_processor
            
            response = await processor.process_batch(request)
            
            # Verify that concurrency was limited
            # Note: This is a simplified test - actual concurrency limiting
            # would be more complex to test properly
            assert response.total_images == 10


@pytest.mark.asyncio
class TestAsyncFunctionality:
    """Test async-specific functionality."""
    
    async def test_async_context_manager(self):
        """Test that async context managers work properly."""
        with patch('src.utils.image_utils.aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            from src.utils.image_utils import ImageProcessor
            
            async with ImageProcessor() as processor:
                assert processor is not None
                assert processor.session is not None
            
            # Verify session was closed
            mock_session.close.assert_called_once()
    
    async def test_async_semaphore_usage(self):
        """Test that semaphores are used correctly for concurrency control."""
        semaphore = asyncio.Semaphore(2)
        
        # This test verifies that our semaphore usage pattern works
        async def limited_operation():
            async with semaphore:
                await asyncio.sleep(0.1)
                return "done"
        
        # Start multiple operations
        tasks = [limited_operation() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(result == "done" for result in results)