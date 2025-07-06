import os
import time
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import threading

from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.core.config import get_settings
from src.models.schemas import Detection
from src.utils.logger import get_logger
from src.utils.metrics import MetricsTimer
from src.utils.brand_classifier import get_brand_classifier

logger = get_logger(__name__)


class DetectionEngineError(Exception):
    """Exception raised by detection engine."""
    pass


class MultiModelDetectionEngine:
    """Multi-model YOLO detection engine with brand classification support."""
    
    def __init__(self, default_model: Optional[str] = None):
        self.settings = get_settings()
        self.current_model_name = default_model or self.settings.default_model
        self.device = self.settings.model_device
        self.models: Dict[str, YOLO] = {}
        self.model_info: Dict[str, Dict] = {}
        self._models_loaded: Dict[str, bool] = {}
        self._lock = threading.Lock()
        self.brand_classifier = get_brand_classifier()
        
        # CPU optimization settings
        self._setup_cpu_optimization()
        
        # Initialize default model
        self._load_model(self.current_model_name)
    
    def _setup_cpu_optimization(self):
        """Setup CPU-specific optimizations."""
        try:
            # Set number of threads for PyTorch
            torch.set_num_threads(min(4, torch.get_num_threads()))
            
            # Disable gradient computation for inference
            torch.set_grad_enabled(False)
            
            # Set CPU optimization flags
            torch.backends.cudnn.enabled = False
            torch.backends.mkldnn.enabled = True
            
            logger.info(f"CPU optimization enabled. Threads: {torch.get_num_threads()}")
            
        except Exception as e:
            logger.warning(f"Failed to setup CPU optimization: {e}")
    
    def _load_model(self, model_name: str):
        """Load a specific YOLO model."""
        try:
            if model_name in self._models_loaded and self._models_loaded[model_name]:
                logger.info(f"Model {model_name} already loaded")
                return
            
            # Get model path - first check configuration, then scan directories
            model_path = None
            
            # Check configuration first
            if model_name in self.settings.available_models:
                model_path = Path(self.settings.available_models[model_name])
            else:
                # Check models directory
                models_dir = Path("models")
                if models_dir.exists():
                    model_file = models_dir / f"{model_name}.pt"
                    if model_file.exists():
                        model_path = model_file
                
                # Check trained models directory
                if model_path is None:
                    trained_model_dir = Path("models/trained") / model_name / "weights" / "best.pt"
                    if trained_model_dir.exists():
                        model_path = trained_model_dir
            
            if model_path is None:
                raise DetectionEngineError(f"Model '{model_name}' not found in available models or directories")
            
            # Check if model file exists
            if not model_path.exists():
                # Try to download default model if it's the general model
                if model_name == "general" and not self._download_default_model(model_path):
                    raise DetectionEngineError(f"Model file not found: {model_path}")
                elif model_name != "general":
                    raise DetectionEngineError(f"Model file not found: {model_path}. Please ensure the trademark model is available.")
            
            logger.info(f"Loading YOLO model '{model_name}' from: {model_path}")
            
            # Load model with CPU device
            model = YOLO(str(model_path))
            model.to(self.device)
            
            # Store model and info
            self.models[model_name] = model
            self.model_info[model_name] = {
                "name": model_name,
                "path": str(model_path),
                "device": self.device,
                "classes": getattr(model, 'names', {}),
                "num_classes": len(getattr(model, 'names', {}))
            }
            
            # Warm up the model with a dummy inference
            self._warmup_model(model_name)
            
            self._models_loaded[model_name] = True
            logger.info(f"YOLO model '{model_name}' loaded successfully")
            
        except Exception as e:
            self._models_loaded[model_name] = False
            raise DetectionEngineError(f"Failed to load model '{model_name}': {str(e)}")
    
    def _download_default_model(self, target_path: Path) -> bool:
        """Download default YOLOv8 model if not present."""
        try:
            logger.info("Downloading default YOLOv8n model...")
            
            # Create models directory if it doesn't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download YOLOv8n model (will be cached by ultralytics)
            model = YOLO("yolov8n.pt")
            
            # The model is already downloaded to ultralytics cache
            # Copy it to our models directory
            import shutil
            from ultralytics.utils import ASSETS
            
            default_model_path = ASSETS / "yolov8n.pt"
            if default_model_path.exists():
                shutil.copy(default_model_path, target_path)
                logger.info(f"Default model saved to: {target_path}")
                return True
            else:
                # Fallback: use the model directly from cache
                logger.info("Using default YOLOv8n model from ultralytics cache")
                return True
                
        except Exception as e:
            logger.error(f"Failed to download default model: {e}")
            return False
    
    def _warmup_model(self, model_name: str):
        """Warm up the model with dummy inference."""
        try:
            if model_name not in self.models:
                return
            
            logger.info(f"Warming up model '{model_name}'...")
            
            # Create dummy image
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Run inference
            with MetricsTimer(f"model_warmup_{model_name}"):
                results = self.models[model_name](dummy_image, verbose=False)
            
            logger.info(f"Model '{model_name}' warmup completed")
            
        except Exception as e:
            logger.warning(f"Model '{model_name}' warmup failed: {e}")
    
    def is_loaded(self, model_name: Optional[str] = None) -> bool:
        """Check if model is loaded and ready."""
        if model_name is None:
            model_name = self.current_model_name
        return self._models_loaded.get(model_name, False) and model_name in self.models
    
    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different model.
        
        Args:
            model_name: Name of the model to switch to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if model is available (either in config or directory)
            available_models = self.get_available_models()
            if model_name not in available_models:
                raise DetectionEngineError(f"Model '{model_name}' not available")
            
            # Load model if not already loaded
            if not self.is_loaded(model_name):
                self._load_model(model_name)
            
            self.current_model_name = model_name
            logger.info(f"Switched to model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch to model '{model_name}': {e}")
            return False
    
    def get_available_models(self) -> Dict[str, Dict]:
        """Get information about all available models."""
        models_info = {}
        
        # First, add configured models
        for model_name, model_path in self.settings.available_models.items():
            models_info[model_name] = {
                "name": model_name,
                "path": model_path,
                "loaded": self.is_loaded(model_name),
                "is_current": model_name == self.current_model_name,
                "confidence_threshold": self.settings.model_confidence_thresholds.get(model_name, 0.8),
                "source": "config"
            }
            
            # Add model-specific info if loaded
            if model_name in self.model_info:
                models_info[model_name].update(self.model_info[model_name])
        
        # Then, scan models directory for .pt files
        models_dir = Path("models")
        if models_dir.exists():
            for model_file in models_dir.glob("*.pt"):
                model_name = model_file.stem
                if model_name not in models_info:
                    models_info[model_name] = {
                        "name": model_name,
                        "path": str(model_file),
                        "loaded": self.is_loaded(model_name),
                        "is_current": model_name == self.current_model_name,
                        "confidence_threshold": self.settings.model_confidence_thresholds.get(model_name, 0.5),
                        "source": "directory"
                    }
                    
                    # Add model-specific info if loaded
                    if model_name in self.model_info:
                        models_info[model_name].update(self.model_info[model_name])
        
        # Also scan trained models directory
        trained_models_dir = Path("models/trained")
        if trained_models_dir.exists():
            for model_dir in trained_models_dir.iterdir():
                if model_dir.is_dir():
                    weights_dir = model_dir / "weights"
                    if weights_dir.exists():
                        best_model = weights_dir / "best.pt"
                        if best_model.exists():
                            model_name = model_dir.name
                            if model_name not in models_info:
                                models_info[model_name] = {
                                    "name": model_name,
                                    "path": str(best_model),
                                    "loaded": self.is_loaded(model_name),
                                    "is_current": model_name == self.current_model_name,
                                    "confidence_threshold": self.settings.model_confidence_thresholds.get(model_name, self.settings.default_confidence_threshold),
                                    "source": "trained"
                                }
                                
                                # Add model-specific info if loaded
                                if model_name in self.model_info:
                                    models_info[model_name].update(self.model_info[model_name])
        
        return models_info
    
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        model_name: Optional[str] = None
    ) -> List[Detection]:
        """
        Detect logos in image.
        
        Args:
            image: Input image as numpy array (RGB format)
            confidence_threshold: Minimum confidence threshold
            max_detections: Maximum number of detections to return
            model_name: Specific model to use (defaults to current model)
            
        Returns:
            List of Detection objects
        """
        model_name = model_name or self.current_model_name
        
        if not self.is_loaded(model_name):
            raise DetectionEngineError(f"Model '{model_name}' not loaded")
        
        # Get model-specific confidence threshold if not provided
        if confidence_threshold is None:
            confidence_threshold = self.settings.model_confidence_thresholds.get(
                model_name, self.settings.confidence_threshold
            )
        
        max_detections = max_detections or self.settings.max_detections
        
        try:
            with self._lock:
                with MetricsTimer(f"yolo_inference_{model_name}"):
                    # Run inference
                    results = self.models[model_name](
                        image,
                        conf=confidence_threshold,
                        verbose=False,
                        device=self.device
                    )
                
                # Process results with brand classification
                detections = self._process_results(
                    results,
                    confidence_threshold,
                    max_detections,
                    model_name
                )
                
                return detections
                
        except Exception as e:
            raise DetectionEngineError(f"Detection failed: {str(e)}")
    
    def _process_results(
        self,
        results: List[Results],
        confidence_threshold: float,
        max_detections: int,
        model_name: str
    ) -> List[Detection]:
        """Process YOLO results into Detection objects with brand classification."""
        detections = []
        
        try:
            for result in results:
                if result.boxes is None:
                    continue
                
                boxes = result.boxes
                
                # Extract data
                if len(boxes) == 0:
                    continue
                
                # Get coordinates, confidences, and classes
                coords = boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                
                # Get class names
                class_names = result.names
                
                # Create Detection objects with enhanced information
                for i in range(len(coords)):
                    conf = float(confidences[i])
                    
                    # Apply confidence threshold
                    if conf < confidence_threshold:
                        continue
                    
                    # Get class name
                    class_id = int(classes[i])
                    raw_logo_name = class_names.get(class_id, f"class_{class_id}")
                    
                    # Apply brand classification if enabled
                    brand_info = None
                    category_info = None
                    adjusted_confidence = conf
                    
                    if self.settings.enable_brand_normalization:
                        brand_info = self.brand_classifier.normalize_brand_name(raw_logo_name)
                        
                        if self.settings.enable_category_classification:
                            category_info = self.brand_classifier.get_brand_category(
                                brand_info["normalized"]
                            )
                            
                            # Adjust confidence based on category
                            adjusted_confidence = self.brand_classifier.adjust_confidence_by_category(
                                brand_info["normalized"], conf
                            )
                    
                    # Create enhanced detection object
                    detection_data = {
                        "logo_name": brand_info["japanese"] if brand_info else raw_logo_name,
                        "confidence": adjusted_confidence,
                        "bbox": coords[i].tolist()
                    }
                    
                    # Add extended information if brand classification is enabled
                    if brand_info:
                        detection_data.update({
                            "brand_info": brand_info,
                            "category_info": category_info,
                            "model_used": model_name,
                            "original_confidence": conf,
                            "raw_detection": raw_logo_name
                        })
                    
                    detection = Detection(**detection_data)
                    detections.append(detection)
                
                # Sort by adjusted confidence (descending)
                detections.sort(key=lambda x: x.confidence, reverse=True)
                
                # Limit number of detections
                if max_detections > 0:
                    detections = detections[:max_detections]
        
        except Exception as e:
            logger.error(f"Error processing results: {e}")
            raise DetectionEngineError(f"Failed to process detection results: {str(e)}")
        
        return detections
    
    def batch_detect(
        self,
        images: List[np.ndarray],
        confidence_threshold: Optional[float] = None,
        max_detections: Optional[int] = None
    ) -> List[List[Detection]]:
        """
        Detect logos in multiple images (batch processing).
        
        Args:
            images: List of input images as numpy arrays
            confidence_threshold: Minimum confidence threshold
            max_detections: Maximum number of detections per image
            
        Returns:
            List of detection lists (one per image)
        """
        if not self.is_loaded():
            raise DetectionEngineError("Model not loaded")
        
        confidence_threshold = confidence_threshold or self.settings.confidence_threshold
        max_detections = max_detections or self.settings.max_detections
        
        try:
            with self._lock:
                with MetricsTimer("yolo_batch_inference"):
                    # Run batch inference
                    results = self.model(
                        images,
                        conf=confidence_threshold,
                        verbose=False,
                        device=self.device
                    )
                
                # Process results for each image
                all_detections = []
                for result in results:
                    detections = self._process_results(
                        [result],
                        confidence_threshold,
                        max_detections
                    )
                    all_detections.append(detections)
                
                return all_detections
                
        except Exception as e:
            raise DetectionEngineError(f"Batch detection failed: {str(e)}")
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict:
        """Get model information."""
        model_name = model_name or self.current_model_name
        
        if not self.is_loaded(model_name):
            return {"loaded": False, "error": f"Model '{model_name}' not loaded"}
        
        try:
            info = {
                "loaded": True,
                "current_model": self.current_model_name,
                "requested_model": model_name,
                "device": self.device,
                "model_type": "YOLOv8",
                "available_models": list(self.settings.available_models.keys()),
                "brand_classification_enabled": self.settings.enable_brand_normalization,
                "category_classification_enabled": self.settings.enable_category_classification
            }
            
            # Add model-specific information
            if model_name in self.model_info:
                info.update(self.model_info[model_name])
            
            return info
            
        except Exception as e:
            return {"loaded": False, "error": str(e)}
    
    def reload_model(self, model_name: Optional[str] = None):
        """Reload a specific model or the current model."""
        model_name = model_name or self.current_model_name
        
        # Remove from loaded models
        if model_name in self._models_loaded:
            self._models_loaded[model_name] = False
        
        if model_name in self.models:
            del self.models[model_name]
        
        if model_name in self.model_info:
            del self.model_info[model_name]
        
        # Reload
        self._load_model(model_name)


# Global detection engine instance
_detection_engine: Optional[MultiModelDetectionEngine] = None
_engine_lock = threading.Lock()


def get_detection_engine() -> MultiModelDetectionEngine:
    """Get the global detection engine instance (singleton)."""
    global _detection_engine
    
    if _detection_engine is None:
        with _engine_lock:
            if _detection_engine is None:
                _detection_engine = MultiModelDetectionEngine()
    
    return _detection_engine


def initialize_detection_engine(default_model: Optional[str] = None) -> MultiModelDetectionEngine:
    """Initialize the detection engine with optional default model."""
    global _detection_engine
    
    with _engine_lock:
        _detection_engine = MultiModelDetectionEngine(default_model)
    
    return _detection_engine