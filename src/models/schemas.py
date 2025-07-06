from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, validator
from datetime import datetime


class ImageInput(BaseModel):
    """Input model for a single image."""
    id: int = Field(..., description="Unique identifier for the image")
    image_url: HttpUrl = Field(..., description="URL of the image to process")
    
    @validator('id')
    def validate_id(cls, v):
        if v is None or v < 0:
            raise ValueError('id must be a non-negative integer')
        return v


class ProcessingOptions(BaseModel):
    """Options for image processing."""
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detections"
    )
    max_detections: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of detections per image"
    )
    model_name: Optional[str] = Field(
        None,
        description="Specific model to use for detection (defaults to current model)"
    )
    enable_brand_normalization: Optional[bool] = Field(
        None,
        description="Enable brand name normalization (defaults to system setting)"
    )
    enable_category_classification: Optional[bool] = Field(
        None,
        description="Enable category classification (defaults to system setting)"
    )


# Brand and Category Information Models
class BrandInfo(BaseModel):
    """Model for brand normalization information."""
    original: str = Field(..., description="Original detected brand name")
    normalized: str = Field(..., description="Normalized brand name")
    japanese: str = Field(..., description="Japanese brand name")
    english: str = Field(..., description="English brand name")
    official_name: str = Field(..., description="Official company name")
    aliases: List[str] = Field(default=[], description="Alternative brand names")


class CategoryInfo(BaseModel):
    """Model for category classification information."""
    category: Dict[str, str] = Field(..., description="Main category information")
    subcategory: Optional[Dict[str, str]] = Field(None, description="Subcategory information")


class Detection(BaseModel):
    """Model for a single logo detection with enhanced information."""
    logo_name: str = Field(..., description="Name of the detected logo (localized)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox: List[float] = Field(
        ...,
        min_items=4,
        max_items=4,
        description="Bounding box coordinates [x1, y1, x2, y2]"
    )
    
    # Extended information (optional, added by brand classification)
    brand_info: Optional[BrandInfo] = Field(None, description="Brand normalization information")
    category_info: Optional[CategoryInfo] = Field(None, description="Category classification information")
    model_used: Optional[str] = Field(None, description="Model used for detection")
    original_confidence: Optional[float] = Field(None, description="Original confidence before adjustment")
    raw_detection: Optional[str] = Field(None, description="Raw model output before normalization")
    
    @validator('bbox')
    def validate_bbox(cls, v):
        if len(v) != 4:
            raise ValueError('bbox must contain exactly 4 coordinates')
        if any(coord < 0 for coord in v):
            raise ValueError('bbox coordinates must be non-negative')
        if v[0] >= v[2] or v[1] >= v[3]:
            raise ValueError('bbox must have valid coordinates (x1 < x2, y1 < y2)')
        return v


class ImageResult(BaseModel):
    """Result for a single image processing."""
    id: int = Field(..., description="Unique identifier for the image")
    detections: List[Detection] = Field(default=[], description="List of detections found")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    status: str = Field(default="success", description="Processing status")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")


class ImageError(BaseModel):
    """Error information for failed image processing."""
    id: int = Field(..., description="Unique identifier for the image")
    error: str = Field(..., description="Error message")


# Batch Processing Models
class BatchProcessingRequest(BaseModel):
    """Request model for batch processing."""
    batch_id: str = Field(..., description="Unique identifier for the batch")
    images: List[ImageInput] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of images to process"
    )
    options: ProcessingOptions = Field(default_factory=ProcessingOptions)
    
    @validator('batch_id')
    def validate_batch_id(cls, v):
        if not v or not v.strip():
            raise ValueError('batch_id cannot be empty')
        return v.strip()
    
    @validator('images')
    def validate_unique_image_ids(cls, v):
        image_ids = [img.id for img in v]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError('All ids must be unique within a batch')
        return v


class BatchProcessingResponse(BaseModel):
    """Response model for batch processing."""
    batch_id: str = Field(..., description="Unique identifier for the batch")
    processing_time: float = Field(..., ge=0, description="Total processing time in seconds")
    total_images: int = Field(..., ge=0, description="Total number of images in batch")
    successful: int = Field(..., ge=0, description="Number of successfully processed images")
    failed: int = Field(..., ge=0, description="Number of failed images")
    results: List[ImageResult] = Field(default=[], description="List of processing results")
    errors: List[ImageError] = Field(default=[], description="List of processing errors")


# Single Image Processing Models
class SingleImageRequest(BaseModel):
    """Request model for single image processing."""
    image_url: HttpUrl = Field(..., description="URL of the image to process")
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detections"
    )
    max_detections: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of detections per image"
    )


class SingleImageResponse(BaseModel):
    """Response model for single image processing."""
    detections: List[Detection] = Field(default=[], description="List of detections found")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    status: str = Field(default="success", description="Processing status")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    image_info: Optional[Dict[str, Any]] = Field(None, description="Image metadata")


# Health Check Models
class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_loaded: bool = Field(..., description="Whether the detection model is loaded")
    system_info: Dict[str, Any] = Field(..., description="System information")


# Metrics Models
class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""
    total_processed: int = Field(..., description="Total number of images processed")
    total_successful: int = Field(..., description="Total number of successful processes")
    total_failed: int = Field(..., description="Total number of failed processes")
    avg_processing_time: float = Field(..., description="Average processing time per image")
    error_rate: float = Field(..., description="Overall error rate")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    active_batches: int = Field(..., description="Number of currently active batches")
    recent_errors: List[Dict[str, Any]] = Field(default=[], description="Recent error information")
    errors_by_type: Dict[str, int] = Field(default={}, description="Error counts by type")
    performance: Optional[Dict[str, Any]] = Field(None, description="System performance metrics")


# Error Response Models
class ErrorResponse(BaseModel):
    """Generic error response model."""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")


class ValidationErrorResponse(BaseModel):
    """Validation error response model."""
    error: str = Field(..., description="Error message")
    error_type: str = Field(default="validation_error")
    details: List[Dict[str, Any]] = Field(..., description="Detailed validation errors")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Batch Status Models (for future use)
class BatchStatusResponse(BaseModel):
    """Response model for batch status check."""
    batch_id: str = Field(..., description="Unique identifier for the batch")
    status: str = Field(..., description="Current batch status")
    total_images: int = Field(..., description="Total number of images in batch")
    successful: int = Field(..., description="Number of successfully processed images")
    failed: int = Field(..., description="Number of failed images")
    processing_time: float = Field(..., description="Current processing time")
    errors: List[ImageError] = Field(default=[], description="List of processing errors")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


# Model Management Models
class ModelSwitchRequest(BaseModel):
    """Request model for switching models."""
    model_name: str = Field(..., description="Name of the model to switch to")


class ModelInfo(BaseModel):
    """Model information."""
    name: str = Field(..., description="Model name")
    path: str = Field(..., description="Model file path")
    loaded: bool = Field(..., description="Whether the model is loaded")
    is_current: bool = Field(..., description="Whether this is the current active model")
    confidence_threshold: float = Field(..., description="Default confidence threshold for this model")
    device: Optional[str] = Field(None, description="Device the model is running on")
    classes: Optional[Dict[str, str]] = Field(None, description="Model class names")
    num_classes: Optional[int] = Field(None, description="Number of classes the model can detect")


class ModelsResponse(BaseModel):
    """Response model for available models."""
    current_model: str = Field(..., description="Currently active model")
    models: Dict[str, ModelInfo] = Field(..., description="Available models information")
    current_model_info: Dict[str, Any] = Field(..., description="Detailed current model information")
    total_models: int = Field(..., description="Total number of available models")
    loaded_models: int = Field(..., description="Number of currently loaded models")


class ModelSwitchResponse(BaseModel):
    """Response model for model switch operation."""
    success: bool = Field(..., description="Whether the switch was successful")
    message: str = Field(..., description="Status message")
    previous_model: str = Field(..., description="Previous active model")
    new_model: str = Field(..., description="New active model")
    model_info: Dict[str, Any] = Field(..., description="New model information")


# Brand Management Models
class BrandListItem(BaseModel):
    """Model for brand list item."""
    key: str = Field(..., description="Brand key/identifier")
    japanese: str = Field(..., description="Japanese brand name")
    english: str = Field(..., description="English brand name")
    official_name: str = Field(..., description="Official company name")
    category: str = Field(..., description="Brand category (Japanese)")
    category_en: str = Field(..., description="Brand category (English)")


class CategoryItem(BaseModel):
    """Model for category item."""
    key: str = Field(..., description="Category key/identifier")
    name: str = Field(..., description="Category name (Japanese)")
    name_en: str = Field(..., description="Category name (English)")
    brand_count: int = Field(..., description="Number of brands in this category")
    subcategories: List[Dict[str, Any]] = Field(default=[], description="Subcategories")


class BrandDetailResponse(BaseModel):
    """Response model for detailed brand information."""
    brand_info: BrandInfo = Field(..., description="Brand normalization information")
    category_info: Optional[CategoryInfo] = Field(None, description="Category information")
    confidence_adjustment: float = Field(..., description="Confidence adjustment factor for this brand")


# Training Pipeline Models
class TrainingRequest(BaseModel):
    """Request model for starting training."""
    model_name: str = Field(..., description="Name for the trained model")
    dataset_name: str = Field(..., description="Dataset to use for training")
    base_model: str = Field("yolov8n.pt", description="Base model to start from")
    epochs: Optional[int] = Field(None, ge=1, le=1000, description="Number of training epochs")
    batch_size: Optional[int] = Field(None, ge=1, le=128, description="Training batch size")
    learning_rate: Optional[float] = Field(None, ge=0.0001, le=1.0, description="Learning rate")
    resume_training: bool = Field(False, description="Resume from existing checkpoint")


class TrainingProgress(BaseModel):
    """Model for training progress information."""
    epoch: int = Field(..., description="Current epoch")
    total_epochs: int = Field(..., description="Total number of epochs")
    loss: float = Field(..., description="Current training loss")
    val_loss: Optional[float] = Field(None, description="Current validation loss")
    mAP: Optional[float] = Field(None, description="Mean Average Precision")
    precision: Optional[float] = Field(None, description="Current precision")
    recall: Optional[float] = Field(None, description="Current recall")
    status: str = Field(..., description="Training status")
    current_step: str = Field(..., description="Current training step description")
    start_time: Optional[datetime] = Field(None, description="Training start time")
    end_time: Optional[datetime] = Field(None, description="Training end time")
    best_mAP: float = Field(0.0, description="Best mAP achieved so far")
    early_stopping_counter: int = Field(0, description="Early stopping patience counter")


class TrainingStatus(BaseModel):
    """Model for training status response."""
    is_training: bool = Field(..., description="Whether training is currently active")
    training_enabled: bool = Field(..., description="Whether training pipeline is enabled")
    progress: Optional[TrainingProgress] = Field(None, description="Current training progress")
    available_models: List[Dict[str, Any]] = Field(default=[], description="Available trained models")
    dataset_info: Optional[Dict[str, Any]] = Field(None, description="Current dataset information")


class DatasetCreateRequest(BaseModel):
    """Request model for dataset creation."""
    name: str = Field(..., min_length=1, max_length=100, description="Dataset name")
    classes: List[str] = Field(..., min_items=1, max_items=50, description="List of class names")
    description: Optional[str] = Field(None, max_length=500, description="Dataset description")


class ImageAnnotation(BaseModel):
    """Model for image annotation."""
    class_name: str = Field(..., description="Class name for the annotation")
    bbox: List[float] = Field(..., min_items=4, max_items=4, description="Bounding box [x_min, y_min, x_max, y_max]")
    confidence: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Annotation confidence")
    
    @validator('bbox')
    def validate_annotation_bbox(cls, v):
        if len(v) != 4:
            raise ValueError('bbox must contain exactly 4 coordinates')
        if any(coord < 0 for coord in v):
            raise ValueError('bbox coordinates must be non-negative')
        if v[0] >= v[2] or v[1] >= v[3]:
            raise ValueError('bbox must have valid coordinates (x1 < x2, y1 < y2)')
        return v


class AddImageRequest(BaseModel):
    """Request model for adding image to dataset."""
    image_path: str = Field(..., description="Path to image file")
    annotations: List[ImageAnnotation] = Field(..., min_items=1, description="List of annotations")
    split: str = Field("train", description="Dataset split (train/val/test)")
    
    @validator('split')
    def validate_split(cls, v):
        if v not in ["train", "val", "test"]:
            raise ValueError('split must be one of: train, val, test')
        return v


class DatasetSplitRequest(BaseModel):
    """Request model for dataset splitting."""
    train_ratio: float = Field(0.7, ge=0.1, le=0.9, description="Training set ratio")
    val_ratio: float = Field(0.2, ge=0.1, le=0.8, description="Validation set ratio")
    test_ratio: float = Field(0.1, ge=0.0, le=0.8, description="Test set ratio")
    
    @validator('test_ratio')
    def validate_ratios_sum(cls, v, values):
        train_ratio = values.get('train_ratio', 0.7)
        val_ratio = values.get('val_ratio', 0.2)
        total = train_ratio + val_ratio + v
        if abs(total - 1.0) > 0.01:
            raise ValueError('train_ratio + val_ratio + test_ratio must equal 1.0')
        return v


class DatasetStats(BaseModel):
    """Model for dataset statistics."""
    total_images: int = Field(..., description="Total number of images")
    total_labels: int = Field(..., description="Total number of label files")
    classes: Dict[str, int] = Field(..., description="Class distribution")
    split_distribution: Dict[str, Dict[str, Any]] = Field(..., description="Distribution across splits")
    avg_image_size: Dict[str, float] = Field(..., description="Average image dimensions")
    avg_annotations_per_image: float = Field(..., description="Average annotations per image")


class DatasetValidation(BaseModel):
    """Model for dataset validation results."""
    is_valid: bool = Field(..., description="Whether dataset is valid for training")
    errors: List[str] = Field(default=[], description="Validation errors")
    warnings: List[str] = Field(default=[], description="Validation warnings")
    summary: Dict[str, Any] = Field(..., description="Validation summary")


class ModelExportRequest(BaseModel):
    """Request model for model export."""
    model_path: str = Field(..., description="Path to model file")
    format: str = Field("onnx", description="Export format (onnx/tensorrt)")
    
    @validator('format')
    def validate_format(cls, v):
        if v.lower() not in ["onnx", "tensorrt"]:
            raise ValueError('format must be either "onnx" or "tensorrt"')
        return v.lower()


class SampleDatasetRequest(BaseModel):
    """Request model for sample dataset generation."""
    dataset_name: str = Field(..., min_length=1, max_length=100, description="Dataset name")
    classes: List[str] = Field(..., min_items=1, max_items=20, description="List of class names")
    images_per_class: int = Field(20, ge=5, le=100, description="Number of images per class")
    image_variations: bool = Field(True, description="Generate image variations")


# Logo Management Models
class LogoClass(BaseModel):
    """Model for logo class information."""
    name: str = Field(..., description="Logo class name")
    description: Optional[str] = Field(None, description="Class description")
    category: Optional[str] = Field(None, description="Logo category")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    created: str = Field(..., description="Creation timestamp")
    image_count: int = Field(0, description="Number of training images")
    is_active: bool = Field(True, description="Whether class is active")


class LogoUploadResponse(BaseModel):
    """Response model for logo upload."""
    success: bool = Field(..., description="Whether upload was successful")
    message: str = Field(..., description="Status message")
    logo_id: str = Field(..., description="Unique logo identifier")
    file_path: str = Field(..., description="Path to uploaded file")
    processed: bool = Field(False, description="Whether image was processed")


class TrainedModelInfo(BaseModel):
    """Model for trained model information."""
    name: str = Field(..., description="Model name")
    path: str = Field(..., description="Model file path")
    size_mb: float = Field(..., description="Model file size in MB")
    created: str = Field(..., description="Creation timestamp")
    version: str = Field(..., description="Model version")
    classes: List[str] = Field(default=[], description="Model classes")
    accuracy_metrics: Optional[Dict[str, float]] = Field(None, description="Accuracy metrics")
    is_deployed: bool = Field(False, description="Whether model is deployed")


class LogoSearchRequest(BaseModel):
    """Request model for logo search."""
    query: str = Field(..., min_length=1, description="Search query")
    category: Optional[str] = Field(None, description="Filter by category")
    limit: int = Field(10, ge=1, le=100, description="Maximum results")


class LogoSimilarityRequest(BaseModel):
    """Request model for logo similarity search."""
    reference_logo: str = Field(..., description="Reference logo name or ID")
    threshold: float = Field(0.8, ge=0.0, le=1.0, description="Similarity threshold")
    limit: int = Field(10, ge=1, le=50, description="Maximum results")


# WebSocket Models
class WebSocketMessage(BaseModel):
    """Model for WebSocket messages."""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")


class TrainingProgressUpdate(BaseModel):
    """Model for training progress WebSocket updates."""
    type: str = Field("progress_update", description="Message type")
    epoch: int = Field(..., description="Current epoch")
    total_epochs: int = Field(..., description="Total epochs")
    metrics: Dict[str, float] = Field(..., description="Current metrics")
    status: str = Field(..., description="Training status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Update timestamp")


# Advanced Training Models
class TrainingHyperparameters(BaseModel):
    """Model for training hyperparameters."""
    learning_rate: float = Field(0.001, ge=0.0001, le=1.0)
    batch_size: int = Field(16, ge=1, le=128)
    epochs: int = Field(50, ge=1, le=1000)
    weight_decay: float = Field(0.0005, ge=0.0, le=0.1)
    momentum: float = Field(0.937, ge=0.0, le=1.0)
    warmup_epochs: int = Field(3, ge=0, le=10)
    patience: int = Field(10, ge=1, le=100)
    
    # Data augmentation parameters
    mosaic: float = Field(1.0, ge=0.0, le=1.0)
    mixup: float = Field(0.0, ge=0.0, le=1.0)
    copy_paste: float = Field(0.0, ge=0.0, le=1.0)
    
    # Image processing parameters
    image_size: int = Field(640, ge=32, le=1920)
    hsv_h: float = Field(0.015, ge=0.0, le=1.0)
    hsv_s: float = Field(0.7, ge=0.0, le=1.0)
    hsv_v: float = Field(0.4, ge=0.0, le=1.0)


class AdvancedTrainingRequest(BaseModel):
    """Advanced training request with detailed configuration."""
    model_name: str = Field(..., description="Name for the trained model")
    dataset_name: str = Field(..., description="Dataset to use for training")
    base_model: str = Field("yolov8n.pt", description="Base model to start from")
    hyperparameters: TrainingHyperparameters = Field(default_factory=TrainingHyperparameters)
    save_checkpoints: bool = Field(True, description="Save training checkpoints")
    early_stopping: bool = Field(True, description="Enable early stopping")
    mixed_precision: bool = Field(True, description="Use mixed precision training")
    resume_checkpoint: Optional[str] = Field(None, description="Checkpoint to resume from")


class TrainingMetrics(BaseModel):
    """Model for detailed training metrics."""
    epoch: int = Field(..., description="Epoch number")
    train_loss: float = Field(..., description="Training loss")
    val_loss: Optional[float] = Field(None, description="Validation loss")
    mAP50: Optional[float] = Field(None, description="mAP at IoU 0.5")
    mAP50_95: Optional[float] = Field(None, description="mAP at IoU 0.5-0.95")
    precision: Optional[float] = Field(None, description="Precision")
    recall: Optional[float] = Field(None, description="Recall")
    learning_rate: float = Field(..., description="Current learning rate")
    gpu_memory: Optional[float] = Field(None, description="GPU memory usage")
    epoch_time: float = Field(..., description="Time taken for this epoch")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metric timestamp")