import os
from typing import Optional, Dict, List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # API Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    environment: str = "prod"  # .envのENVIRONMENTフィールド対応
    
    # Model Configuration
    model_path: str = "models/yolov8n.pt"
    model_device: str = "cpu"
    confidence_threshold: float = 0.8
    max_detections: int = 10
    
    # Multi-Model Configuration
    default_model: str = "general"
    available_models: Dict[str, str] = {
        "general": "models/yolov8n.pt",
        "trademark": "models/trademark_logos.pt",
        "custom": "models/custom_logos.pt"
    }
    
    # Model-specific confidence thresholds
    model_confidence_thresholds: Dict[str, float] = {
        "general": 0.8,
        "trademark": 0.7,
        "custom": 0.75
    }
    # Default confidence threshold for dynamically loaded models
    default_confidence_threshold: float = 0.7
    
    # Logo Classification Configuration
    enable_brand_normalization: bool = True
    enable_category_classification: bool = True
    logo_categories_file: str = "data/logo_categories.json"
    brand_mapping_file: str = "data/brand_mapping.json"
    
    # Batch Processing Configuration
    max_batch_size: int = 100
    max_concurrent_downloads: int = 50
    max_concurrent_detections: int = 10
    download_timeout: int = 30
    processing_timeout: int = 300
    
    # Image Processing Configuration
    max_image_size: int = 1920
    supported_formats: list[str] = ["jpg", "jpeg", "png", "bmp", "webp"]
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    
    # Performance Monitoring
    enable_metrics: bool = True
    metrics_retention_hours: int = 24
    
    # Security
    cors_origins: list[str] = ["*"]
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    
    # Training Pipeline Configuration
    training_enabled: bool = True
    training_data_dir: str = "datasets/"
    training_output_dir: str = "models/trained/"
    default_epochs: int = 50
    default_batch_size: int = 16
    default_learning_rate: float = 0.001
    initial_logo_classes: List[str] = ["BANDAI", "Nintendo", "KONAMI", "SONY", "Panasonic"]
    
    # Training Dataset Configuration
    min_images_per_class: int = 5
    max_images_per_class: int = 100
    train_val_split: float = 0.8
    augmentation_enabled: bool = True
    
    # Training Monitoring
    training_log_dir: str = "logs/training/"
    save_training_checkpoints: bool = True
    checkpoint_frequency: int = 10
    early_stopping_patience: int = 10
    
    # Model Export Configuration
    export_onnx: bool = True
    export_tensorrt: bool = False
    quantization_enabled: bool = False

    # class Config 部分は削除


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def update_settings(**kwargs) -> None:
    """Update settings dynamically."""
    global settings
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)


# Environment-specific configurations
def is_development() -> bool:
    """Check if running in development mode."""
    return settings.debug or os.getenv("ENVIRONMENT", "dev").lower() == "dev"


def is_production() -> bool:
    """Check if running in production mode."""
    return os.getenv("ENVIRONMENT", "dev").lower() == "prod"