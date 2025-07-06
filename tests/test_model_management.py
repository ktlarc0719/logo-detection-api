import pytest
import asyncio
from httpx import AsyncClient
from unittest.mock import patch, MagicMock, AsyncMock

from src.api.main import create_app
from src.models.schemas import BrandInfo, CategoryInfo, Detection


@pytest.fixture
def app():
    """Create test app instance."""
    return create_app()


@pytest.fixture
async def async_client(app):
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_detection_engine():
    """Mock detection engine with multi-model support."""
    mock_engine = MagicMock()
    mock_engine.current_model_name = "general"
    mock_engine.get_available_models.return_value = {
        "general": {
            "name": "general",
            "path": "models/yolov8n.pt",
            "loaded": True,
            "is_current": True,
            "confidence_threshold": 0.8
        },
        "trademark": {
            "name": "trademark",
            "path": "models/trademark_logos.pt",
            "loaded": False,
            "is_current": False,
            "confidence_threshold": 0.7
        }
    }
    mock_engine.get_model_info.return_value = {
        "loaded": True,
        "current_model": "general",
        "device": "cpu",
        "model_type": "YOLOv8",
        "available_models": ["general", "trademark"],
        "brand_classification_enabled": True,
        "category_classification_enabled": True
    }
    mock_engine.switch_model.return_value = True
    mock_engine.is_loaded.return_value = True
    return mock_engine


@pytest.fixture
def mock_brand_classifier():
    """Mock brand classifier."""
    mock_classifier = MagicMock()
    mock_classifier.get_available_brands.return_value = [
        {
            "key": "BANDAI",
            "japanese": "バンダイ",
            "english": "BANDAI",
            "official_name": "株式会社バンダイ",
            "category": "玩具・ゲーム",
            "category_en": "Toys & Games"
        }
    ]
    mock_classifier.get_available_categories.return_value = [
        {
            "key": "toys_games",
            "name": "玩具・ゲーム",
            "name_en": "Toys & Games",
            "brand_count": 8,
            "subcategories": []
        }
    ]
    mock_classifier.normalize_brand_name.return_value = {
        "original": "BANDAI",
        "normalized": "BANDAI",
        "japanese": "バンダイ",
        "english": "BANDAI",
        "official_name": "株式会社バンダイ",
        "aliases": ["BANDAI", "バンダイ"]
    }
    mock_classifier.get_brand_category.return_value = {
        "category": {
            "key": "toys_games",
            "name": "玩具・ゲーム",
            "name_en": "Toys & Games"
        },
        "subcategory": None
    }
    mock_classifier.adjust_confidence_by_category.return_value = 0.75
    return mock_classifier


class TestModelManagementAPI:
    """Test model management endpoints."""
    
    @patch('src.api.endpoints.model_management.get_detection_engine')
    async def test_get_available_models(self, mock_get_engine, async_client, mock_detection_engine):
        """Test get available models endpoint."""
        mock_get_engine.return_value = mock_detection_engine
        
        response = await async_client.get("/api/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "current_model" in data
        assert "models" in data
        assert "total_models" in data
        assert data["current_model"] == "general"
        assert len(data["models"]) == 2
    
    @patch('src.api.endpoints.model_management.get_detection_engine')
    async def test_get_current_model(self, mock_get_engine, async_client, mock_detection_engine):
        """Test get current model endpoint."""
        mock_get_engine.return_value = mock_detection_engine
        
        response = await async_client.get("/api/v1/models/current")
        assert response.status_code == 200
        
        data = response.json()
        assert data["loaded"] is True
        assert data["current_model"] == "general"
        assert data["brand_classification_enabled"] is True
    
    @patch('src.api.endpoints.model_management.get_detection_engine')
    async def test_switch_model_success(self, mock_get_engine, async_client, mock_detection_engine):
        """Test successful model switch."""
        mock_get_engine.return_value = mock_detection_engine
        
        response = await async_client.post("/api/v1/models/switch?model_name=trademark")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["new_model"] == "trademark"
        assert "model_info" in data
    
    @patch('src.api.endpoints.model_management.get_detection_engine')
    async def test_switch_model_invalid(self, mock_get_engine, async_client, mock_detection_engine):
        """Test model switch with invalid model name."""
        mock_get_engine.return_value = mock_detection_engine
        
        response = await async_client.post("/api/v1/models/switch?model_name=nonexistent")
        assert response.status_code == 400
        assert "not available" in response.json()["detail"]
    
    @patch('src.api.endpoints.model_management.get_detection_engine')
    async def test_load_model(self, mock_get_engine, async_client, mock_detection_engine):
        """Test model loading endpoint."""
        mock_get_engine.return_value = mock_detection_engine
        mock_detection_engine.is_loaded.return_value = False
        
        response = await async_client.post("/api/v1/models/trademark/load")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["model_name"] == "trademark"


class TestBrandManagementAPI:
    """Test brand management endpoints."""
    
    @patch('src.api.endpoints.model_management.get_brand_classifier')
    async def test_get_available_brands(self, mock_get_classifier, async_client, mock_brand_classifier):
        """Test get available brands endpoint."""
        mock_get_classifier.return_value = mock_brand_classifier
        
        response = await async_client.get("/api/v1/brands")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 1
        assert data[0]["key"] == "BANDAI"
        assert data[0]["japanese"] == "バンダイ"
    
    @patch('src.api.endpoints.model_management.get_brand_classifier')
    async def test_get_brand_categories(self, mock_get_classifier, async_client, mock_brand_classifier):
        """Test get brand categories endpoint."""
        mock_get_classifier.return_value = mock_brand_classifier
        
        response = await async_client.get("/api/v1/categories")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 1
        assert data[0]["key"] == "toys_games"
        assert data[0]["name"] == "玩具・ゲーム"
    
    @patch('src.api.endpoints.model_management.get_brand_classifier')
    async def test_get_brand_info(self, mock_get_classifier, async_client, mock_brand_classifier):
        """Test get brand information endpoint."""
        mock_get_classifier.return_value = mock_brand_classifier
        
        response = await async_client.get("/api/v1/brands/BANDAI/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "brand_info" in data
        assert "category_info" in data
        assert "confidence_adjustment" in data
        assert data["brand_info"]["japanese"] == "バンダイ"
    
    @patch('src.api.endpoints.model_management.get_brand_classifier')
    async def test_reload_brand_data(self, mock_get_classifier, async_client, mock_brand_classifier):
        """Test reload brand data endpoint."""
        mock_get_classifier.return_value = mock_brand_classifier
        
        response = await async_client.post("/api/v1/brands/reload")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        mock_brand_classifier.reload_data.assert_called_once()


class TestEnhancedDetection:
    """Test enhanced detection with brand classification."""
    
    def test_enhanced_detection_creation(self):
        """Test creation of enhanced detection object."""
        brand_info = BrandInfo(
            original="BANDAI",
            normalized="BANDAI",
            japanese="バンダイ",
            english="BANDAI",
            official_name="株式会社バンダイ",
            aliases=["BANDAI", "バンダイ"]
        )
        
        category_info = CategoryInfo(
            category={"key": "toys_games", "name": "玩具・ゲーム"},
            subcategory=None
        )
        
        detection = Detection(
            logo_name="バンダイ",
            confidence=0.95,
            bbox=[100, 50, 200, 100],
            brand_info=brand_info,
            category_info=category_info,
            model_used="trademark",
            original_confidence=0.90,
            raw_detection="BANDAI"
        )
        
        assert detection.logo_name == "バンダイ"
        assert detection.confidence == 0.95
        assert detection.brand_info.japanese == "バンダイ"
        assert detection.category_info.category["name"] == "玩具・ゲーム"
        assert detection.model_used == "trademark"
        assert detection.original_confidence == 0.90
    
    def test_basic_detection_still_works(self):
        """Test that basic detection objects still work without extended info."""
        detection = Detection(
            logo_name="BANDAI",
            confidence=0.95,
            bbox=[100, 50, 200, 100]
        )
        
        assert detection.logo_name == "BANDAI"
        assert detection.confidence == 0.95
        assert detection.bbox == [100, 50, 200, 100]
        assert detection.brand_info is None
        assert detection.category_info is None


class TestBrandClassification:
    """Test brand classification functionality."""
    
    @patch('src.utils.brand_classifier.Path')
    @patch('builtins.open')
    @patch('json.load')
    def test_brand_classifier_initialization(self, mock_json_load, mock_open, mock_path):
        """Test brand classifier initialization."""
        from src.utils.brand_classifier import BrandClassifier
        
        # Mock file existence and content
        mock_path.return_value.exists.return_value = True
        mock_json_load.side_effect = [
            {"brand_normalization": {"BANDAI": {"japanese": "バンダイ"}}},
            {"categories": {"toys_games": {"name": "玩具・ゲーム"}}}
        ]
        
        classifier = BrandClassifier()
        
        assert "brand_normalization" in classifier.brand_mapping
        assert "categories" in classifier.logo_categories
    
    def test_brand_name_normalization(self):
        """Test brand name normalization logic."""
        from src.utils.brand_classifier import BrandClassifier
        
        classifier = BrandClassifier()
        classifier.brand_mapping = {
            "brand_normalization": {
                "BANDAI": {
                    "japanese": "バンダイ",
                    "english": "BANDAI",
                    "official_name": "株式会社バンダイ",
                    "aliases": ["BANDAI", "バンダイ"]
                }
            }
        }
        
        result = classifier.normalize_brand_name("BANDAI")
        
        assert result["original"] == "BANDAI"
        assert result["normalized"] == "BANDAI"
        assert result["japanese"] == "バンダイ"
        assert result["official_name"] == "株式会社バンダイ"
    
    def test_confidence_adjustment(self):
        """Test confidence adjustment by category."""
        from src.utils.brand_classifier import BrandClassifier
        
        classifier = BrandClassifier()
        classifier.logo_categories = {
            "categories": {
                "toys_games": {
                    "brands": ["BANDAI"]
                }
            },
            "confidence_adjustments": {
                "toys_games": {
                    "threshold_adjustment": -0.05
                }
            }
        }
        
        adjusted = classifier.adjust_confidence_by_category("BANDAI", 0.8)
        assert adjusted == 0.75  # 0.8 - 0.05


class TestProcessingOptions:
    """Test extended processing options."""
    
    async def test_processing_options_with_model_selection(self):
        """Test processing options include model selection."""
        from src.models.schemas import ProcessingOptions
        
        options = ProcessingOptions(
            confidence_threshold=0.7,
            max_detections=5,
            model_name="trademark",
            enable_brand_normalization=True,
            enable_category_classification=True
        )
        
        assert options.confidence_threshold == 0.7
        assert options.max_detections == 5
        assert options.model_name == "trademark"
        assert options.enable_brand_normalization is True
        assert options.enable_category_classification is True


@pytest.mark.integration
class TestIntegrationWithEnhancedFeatures:
    """Integration tests for enhanced features."""
    
    @patch('src.api.endpoints.single_detection.get_batch_processor')
    async def test_single_image_with_enhanced_detection(self, mock_get_processor, async_client):
        """Test single image processing with enhanced detection features."""
        # Mock enhanced detection result
        mock_result = MagicMock()
        mock_result.detections = [
            Detection(
                logo_name="バンダイ",
                confidence=0.95,
                bbox=[100, 50, 200, 100],
                brand_info=BrandInfo(
                    original="BANDAI",
                    normalized="BANDAI",
                    japanese="バンダイ",
                    english="BANDAI",
                    official_name="株式会社バンダイ",
                    aliases=["BANDAI", "バンダイ"]
                ),
                model_used="trademark",
                original_confidence=0.90
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
            "confidence_threshold": 0.7,
            "max_detections": 10
        }
        
        response = await async_client.post("/api/v1/process/single", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["detections"]) == 1
        detection = data["detections"][0]
        assert detection["logo_name"] == "バンダイ"
        assert "brand_info" in detection
        assert detection["brand_info"]["japanese"] == "バンダイ"
        assert detection["model_used"] == "trademark"