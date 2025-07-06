import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

from src.api.main import create_app
from src.models.schemas import (
    BatchProcessingRequest, SingleImageRequest, ImageInput, ProcessingOptions
)


@pytest.fixture
def app():
    """Create test app instance."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
async def async_client(app):
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_detection_engine():
    """Mock detection engine."""
    mock_engine = MagicMock()
    mock_engine.is_loaded.return_value = True
    mock_engine.get_model_info.return_value = {
        "loaded": True,
        "model_path": "models/yolov8n.pt",
        "device": "cpu",
        "model_type": "YOLOv8",
        "classes": {"0": "logo"},
        "num_classes": 1
    }
    return mock_engine


@pytest.fixture
def mock_batch_processor():
    """Mock batch processor."""
    mock_processor = MagicMock()
    mock_processor.get_status.return_value = {
        "detection_engine_loaded": True,
        "max_concurrent_downloads": 50,
        "max_concurrent_detections": 10,
        "max_batch_size": 100,
        "active_threads": 0,
        "pending_tasks": 0
    }
    return mock_processor


class TestHealthEndpoints:
    """Test health and monitoring endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Logo Detection API"
        assert "endpoints" in data
    
    @patch('src.api.endpoints.health.get_detection_engine')
    def test_health_check_success(self, mock_get_engine, client, mock_detection_engine):
        """Test successful health check."""
        mock_get_engine.return_value = mock_detection_engine
        
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert "system_info" in data
    
    @patch('src.api.endpoints.health.get_detection_engine')
    def test_health_check_degraded(self, mock_get_engine, client):
        """Test health check when model not loaded."""
        mock_engine = MagicMock()
        mock_engine.is_loaded.return_value = False
        mock_get_engine.return_value = mock_engine
        
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False
    
    @patch('src.api.endpoints.health.get_metrics_collector')
    def test_metrics_endpoint(self, mock_get_collector, client):
        """Test metrics endpoint."""
        mock_collector = MagicMock()
        mock_collector.get_current_metrics.return_value = {
            "total_processed": 100,
            "total_successful": 95,
            "total_failed": 5,
            "avg_processing_time": 0.045,
            "error_rate": 0.05,
            "uptime_seconds": 3600,
            "active_batches": 0,
            "recent_errors": [],
            "errors_by_type": {},
            "performance": None
        }
        mock_get_collector.return_value = mock_collector
        
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 100
        assert data["total_successful"] == 95
        assert data["error_rate"] == 0.05
    
    @patch('src.api.endpoints.health.get_detection_engine')
    def test_readiness_probe_ready(self, mock_get_engine, client, mock_detection_engine):
        """Test readiness probe when ready."""
        mock_get_engine.return_value = mock_detection_engine
        
        response = client.get("/api/v1/readiness")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
    
    @patch('src.api.endpoints.health.get_detection_engine')
    def test_readiness_probe_not_ready(self, mock_get_engine, client):
        """Test readiness probe when not ready."""
        mock_engine = MagicMock()
        mock_engine.is_loaded.return_value = False
        mock_get_engine.return_value = mock_engine
        
        response = client.get("/api/v1/readiness")
        assert response.status_code == 503
    
    def test_liveness_probe(self, client):
        """Test liveness probe."""
        response = client.get("/api/v1/liveness")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"


class TestSingleImageEndpoint:
    """Test single image processing endpoint."""
    
    @patch('src.api.endpoints.single_detection.get_batch_processor')
    async def test_single_image_success(self, mock_get_processor, async_client):
        """Test successful single image processing."""
        # Mock successful processing result
        mock_result = MagicMock()
        mock_result.detections = [
            MagicMock(
                logo_name="test_logo",
                confidence=0.95,
                bbox=[100, 50, 200, 100]
            )
        ]
        mock_result.processing_time = 0.045
        mock_result.status = "success"
        mock_result.error_message = None
        
        mock_processor = AsyncMock()
        mock_processor.process_single_image.return_value = mock_result
        mock_get_processor.return_value = mock_processor
        
        request_data = {
            "image_url": "https://example.com/image.jpg",
            "confidence_threshold": 0.8,
            "max_detections": 10
        }
        
        response = await async_client.post("/api/v1/process/single", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert len(data["detections"]) == 1
        assert data["detections"][0]["logo_name"] == "test_logo"
        assert data["processing_time"] == 0.045
    
    async def test_single_image_invalid_url(self, async_client):
        """Test single image processing with invalid URL."""
        request_data = {
            "image_url": "not-a-valid-url",
            "confidence_threshold": 0.8
        }
        
        response = await async_client.post("/api/v1/process/single", json=request_data)
        assert response.status_code == 422  # Validation error
    
    async def test_single_image_missing_url(self, async_client):
        """Test single image processing with missing URL."""
        request_data = {
            "confidence_threshold": 0.8
        }
        
        response = await async_client.post("/api/v1/process/single", json=request_data)
        assert response.status_code == 422  # Validation error


class TestBatchProcessingEndpoint:
    """Test batch processing endpoint."""
    
    @patch('src.api.endpoints.batch_detection.get_batch_processor')
    async def test_batch_processing_success(self, mock_get_processor, async_client):
        """Test successful batch processing."""
        # Mock successful batch processing result
        mock_response = MagicMock()
        mock_response.batch_id = "test_batch_001"
        mock_response.processing_time = 2.34
        mock_response.total_images = 2
        mock_response.successful = 2
        mock_response.failed = 0
        mock_response.results = [
            MagicMock(
                image_id="img_001",
                detections=[],
                processing_time=1.0,
                status="success"
            ),
            MagicMock(
                image_id="img_002",
                detections=[],
                processing_time=1.2,
                status="success"
            )
        ]
        mock_response.errors = []
        
        mock_processor = AsyncMock()
        mock_processor.process_batch.return_value = mock_response
        mock_get_processor.return_value = mock_processor
        
        request_data = {
            "batch_id": "test_batch_001",
            "images": [
                {
                    "image_id": "img_001",
                    "image_url": "https://example.com/image1.jpg"
                },
                {
                    "image_id": "img_002",
                    "image_url": "https://example.com/image2.jpg"
                }
            ],
            "options": {
                "confidence_threshold": 0.8,
                "max_detections": 10
            }
        }
        
        response = await async_client.post("/api/v1/process/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["batch_id"] == "test_batch_001"
        assert data["total_images"] == 2
        assert data["successful"] == 2
        assert data["failed"] == 0
    
    async def test_batch_processing_empty_batch(self, async_client):
        """Test batch processing with empty batch."""
        request_data = {
            "batch_id": "empty_batch",
            "images": [],
            "options": {
                "confidence_threshold": 0.8
            }
        }
        
        response = await async_client.post("/api/v1/process/batch", json=request_data)
        assert response.status_code == 422  # Validation error
    
    async def test_batch_processing_duplicate_image_ids(self, async_client):
        """Test batch processing with duplicate image IDs."""
        request_data = {
            "batch_id": "duplicate_batch",
            "images": [
                {
                    "image_id": "img_001",
                    "image_url": "https://example.com/image1.jpg"
                },
                {
                    "image_id": "img_001",  # Duplicate ID
                    "image_url": "https://example.com/image2.jpg"
                }
            ]
        }
        
        response = await async_client.post("/api/v1/process/batch", json=request_data)
        assert response.status_code == 422  # Validation error
    
    @patch('src.api.endpoints.batch_detection.get_batch_processor')
    async def test_batch_status_endpoint(self, mock_get_processor, async_client, mock_batch_processor):
        """Test batch status endpoint."""
        mock_get_processor.return_value = mock_batch_processor
        
        response = await async_client.get("/api/v1/process/batch/test_batch_001/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["batch_id"] == "test_batch_001"
        assert "processor_status" in data


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    async def test_invalid_json(self, async_client):
        """Test handling of invalid JSON."""
        response = await async_client.post(
            "/api/v1/process/single",
            content="invalid json"
        )
        assert response.status_code == 422
    
    async def test_unsupported_method(self, async_client):
        """Test unsupported HTTP method."""
        response = await async_client.put("/api/v1/process/single")
        assert response.status_code == 405
    
    async def test_nonexistent_endpoint(self, async_client):
        """Test nonexistent endpoint."""
        response = await async_client.get("/api/v1/nonexistent")
        assert response.status_code == 404


@pytest.mark.parametrize("confidence_threshold,max_detections", [
    (0.5, 5),
    (0.9, 20),
    (1.0, 1)
])
class TestParameterValidation:
    """Test parameter validation with different values."""
    
    async def test_single_image_parameters(self, async_client, confidence_threshold, max_detections):
        """Test single image endpoint with different parameters."""
        with patch('src.api.endpoints.single_detection.get_batch_processor') as mock_get_processor:
            mock_result = MagicMock()
            mock_result.detections = []
            mock_result.processing_time = 0.045
            mock_result.status = "success"
            mock_result.error_message = None
            
            mock_processor = AsyncMock()
            mock_processor.process_single_image.return_value = mock_result
            mock_get_processor.return_value = mock_processor
            
            request_data = {
                "image_url": "https://example.com/image.jpg",
                "confidence_threshold": confidence_threshold,
                "max_detections": max_detections
            }
            
            response = await async_client.post("/api/v1/process/single", json=request_data)
            assert response.status_code == 200