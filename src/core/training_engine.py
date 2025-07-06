"""
Training Engine for Logo Detection API

This module provides YOLO model training capabilities for trademark logo detection.
It includes dataset management, model training, progress monitoring, and model export functionality.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER as yolo_logger

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingProgress:
    """Training progress tracker."""
    
    def __init__(self):
        self.epoch = 0
        self.total_epochs = 0
        self.loss = 0.0
        self.val_loss = 0.0
        self.mAP = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.status = "preparing"  # preparing, training, validating, completed, failed
        self.current_step = ""
        self.start_time = None
        self.end_time = None
        self.best_mAP = 0.0
        self.early_stopping_counter = 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "loss": self.loss,
            "val_loss": self.val_loss,
            "mAP": self.mAP,
            "precision": self.precision,
            "recall": self.recall,
            "status": self.status,
            "current_step": self.current_step,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "best_mAP": self.best_mAP,
            "early_stopping_counter": self.early_stopping_counter
        }


class TrainingEngine:
    """YOLO training engine for logo detection."""
    
    def __init__(self):
        self.settings = get_settings()
        self.progress = TrainingProgress()
        self.model = None
        self.training_task = None
        self.progress_callbacks: List[Callable] = []
        
        # Set up directories
        self._setup_directories()
        
        # Configure YOLO logging
        yolo_logger.setLevel(logging.WARNING)
        
    def _setup_directories(self):
        """Create necessary directories for training."""
        directories = [
            self.settings.training_data_dir,
            self.settings.training_output_dir,
            self.settings.training_log_dir,
            os.path.join(self.settings.training_data_dir, "train"),
            os.path.join(self.settings.training_data_dir, "val"),
            os.path.join(self.settings.training_data_dir, "test")
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Training directories created: {directories}")
    
    def add_progress_callback(self, callback: Callable[[TrainingProgress], None]):
        """Add a progress callback function."""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self):
        """Notify all progress callbacks."""
        for callback in self.progress_callbacks:
            try:
                callback(self.progress)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "is_training": self.training_task is not None and not self.training_task.done(),
            "progress": self.progress.to_dict(),
            "available_models": self._get_available_trained_models(),
            "dataset_info": self._get_dataset_info()
        }
    
    def _get_available_trained_models(self) -> List[Dict[str, Any]]:
        """Get list of available trained models."""
        models = []
        output_dir = Path(self.settings.training_output_dir)
        
        if not output_dir.exists():
            return models
        
        for model_path in output_dir.glob("*.pt"):
            try:
                # Get model info
                model_info = {
                    "name": model_path.stem,
                    "path": str(model_path),
                    "size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
                    "created": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(),
                    "is_best": model_path.name.endswith("_best.pt")
                }
                
                # Try to get training metadata
                metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        model_info.update(metadata)
                
                models.append(model_info)
            except Exception as e:
                logger.warning(f"Error reading model info for {model_path}: {e}")
        
        return sorted(models, key=lambda x: x["created"], reverse=True)
    
    def _get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        dataset_dir = Path(self.settings.training_data_dir)
        
        if not dataset_dir.exists():
            return {"exists": False}
        
        info = {"exists": True, "splits": {}}
        
        for split in ["train", "val", "test"]:
            split_dir = dataset_dir / split
            if split_dir.exists():
                # Count images and labels
                image_count = len(list(split_dir.glob("images/*.jpg"))) + len(list(split_dir.glob("images/*.png")))
                label_count = len(list(split_dir.glob("labels/*.txt")))
                
                info["splits"][split] = {
                    "images": image_count,
                    "labels": label_count,
                    "path": str(split_dir)
                }
        
        # Get class information
        classes_file = dataset_dir / "classes.yaml"
        if classes_file.exists():
            try:
                with open(classes_file, 'r') as f:
                    classes_data = yaml.safe_load(f)
                    info["classes"] = classes_data
            except Exception as e:
                logger.warning(f"Error reading classes file: {e}")
        
        return info
    
    async def prepare_dataset(self, 
                             class_names: List[str],
                             images_per_class: Optional[int] = None) -> Dict[str, Any]:
        """Prepare dataset for training."""
        if images_per_class is None:
            images_per_class = self.settings.min_images_per_class
        
        self.progress.status = "preparing"
        self.progress.current_step = "Preparing dataset"
        self._notify_progress()
        
        try:
            # Create dataset structure
            dataset_dir = Path(self.settings.training_data_dir)
            
            # Create YOLO dataset structure
            for split in ["train", "val", "test"]:
                (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
                (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)
            
            # Create classes.yaml file
            classes_yaml = {
                "nc": len(class_names),
                "names": class_names,
                "path": str(dataset_dir.absolute()),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images"
            }
            
            with open(dataset_dir / "classes.yaml", 'w') as f:
                yaml.dump(classes_yaml, f)
            
            logger.info(f"Dataset prepared for classes: {class_names}")
            
            return {
                "success": True,
                "classes": class_names,
                "dataset_path": str(dataset_dir),
                "images_per_class": images_per_class
            }
            
        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            self.progress.status = "failed"
            self.progress.current_step = f"Dataset preparation failed: {str(e)}"
            self._notify_progress()
            return {"success": False, "error": str(e)}
    
    async def start_training(self,
                           model_name: str,
                           dataset_name: Optional[str] = None,
                           base_model: str = "yolov8n.pt",
                           epochs: Optional[int] = None,
                           batch_size: Optional[int] = None,
                           learning_rate: Optional[float] = None) -> Dict[str, Any]:
        """Start model training."""
        if self.training_task and not self.training_task.done():
            return {"success": False, "error": "Training already in progress"}
        
        if epochs is None:
            epochs = self.settings.default_epochs
        if batch_size is None:
            batch_size = self.settings.default_batch_size
        if learning_rate is None:
            learning_rate = self.settings.default_learning_rate
        
        # Check if dataset exists
        if dataset_name:
            # Use specific dataset
            dataset_yaml = Path(self.settings.training_data_dir) / dataset_name / "dataset.yaml"
        else:
            # Legacy path
            dataset_yaml = Path(self.settings.training_data_dir) / "classes.yaml"
            
        if not dataset_yaml.exists():
            return {"success": False, "error": f"Dataset not found at {dataset_yaml}"}
        
        # Start training in background
        self.training_task = asyncio.create_task(
            self._train_model(model_name, dataset_name, base_model, epochs, batch_size, learning_rate)
        )
        
        return {
            "success": True,
            "message": f"Training started for model '{model_name}'",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }
    
    async def _train_model(self,
                          model_name: str,
                          dataset_name: Optional[str],
                          base_model: str,
                          epochs: int,
                          batch_size: int,
                          learning_rate: float) -> Dict[str, Any]:
        """Internal training method."""
        self.progress.status = "training"
        self.progress.epoch = 0
        self.progress.total_epochs = epochs
        self.progress.start_time = datetime.now()
        self.progress.current_step = "Initializing training"
        self._notify_progress()
        
        try:
            # Load base model
            self.model = YOLO(base_model)
            
            # Configure training parameters
            if dataset_name:
                dataset_yaml = Path(self.settings.training_data_dir) / dataset_name / "dataset.yaml"
            else:
                dataset_yaml = Path(self.settings.training_data_dir) / "classes.yaml"
            output_dir = Path(self.settings.training_output_dir)
            
            training_args = {
                "data": str(dataset_yaml),
                "epochs": epochs,
                "batch": batch_size,
                "lr0": learning_rate,
                "project": str(output_dir),
                "name": model_name,
                "save": True,
                "save_period": self.settings.checkpoint_frequency,
                "patience": self.settings.early_stopping_patience,
                "device": "cpu",  # Force CPU for consistency
                "workers": min(4, os.cpu_count() or 1),
                "verbose": True
            }
            
            # Start training
            logger.info(f"Starting training with args: {training_args}")
            
            # Run training in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = loop.run_in_executor(
                    executor,
                    lambda: self.model.train(**training_args)
                )
                
                # Monitor training progress
                while not future.done():
                    await asyncio.sleep(1)
                    # Update progress based on training logs
                    # This is a simplified version - in practice, you'd parse YOLO logs
                    if self.progress.epoch < epochs:
                        self.progress.current_step = f"Training epoch {self.progress.epoch + 1}/{epochs}"
                        self._notify_progress()
                
                # Get training results
                results = await future
            
            # Training completed
            self.progress.status = "completed"
            self.progress.end_time = datetime.now()
            self.progress.current_step = "Training completed"
            self._notify_progress()
            
            # Save model with correct name
            trained_model_path = output_dir / model_name / "weights" / "best.pt"
            if trained_model_path.exists():
                # Save with the model name
                model_save_path = Path("models") / f"{model_name}.pt"
                model_save_path.parent.mkdir(exist_ok=True)
                shutil.copy2(trained_model_path, model_save_path)
                logger.info(f"Model saved to {model_save_path}")
            
            # Save training metadata
            metadata = {
                "model_name": model_name,
                "base_model": base_model,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "training_time": (self.progress.end_time - self.progress.start_time).total_seconds(),
                "final_mAP": self.progress.mAP,
                "created": self.progress.end_time.isoformat()
            }
            
            metadata_path = output_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                "success": True,
                "model_path": str(trained_model_path),
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.progress.status = "failed"
            self.progress.end_time = datetime.now()
            self.progress.current_step = f"Training failed: {str(e)}"
            self._notify_progress()
            return {"success": False, "error": str(e)}
    
    async def stop_training(self) -> Dict[str, Any]:
        """Stop current training."""
        if not self.training_task or self.training_task.done():
            return {"success": False, "error": "No training in progress"}
        
        self.training_task.cancel()
        self.progress.status = "stopped"
        self.progress.end_time = datetime.now()
        self.progress.current_step = "Training stopped by user"
        self._notify_progress()
        
        return {"success": True, "message": "Training stopped"}
    
    def export_model(self, 
                    model_path: str,
                    format: str = "onnx") -> Dict[str, Any]:
        """Export trained model to different formats."""
        try:
            model = YOLO(model_path)
            
            if format.lower() == "onnx" and self.settings.export_onnx:
                export_path = model.export(format="onnx")
                return {"success": True, "export_path": export_path}
            elif format.lower() == "tensorrt" and self.settings.export_tensorrt:
                export_path = model.export(format="engine")
                return {"success": True, "export_path": export_path}
            else:
                return {"success": False, "error": f"Export format '{format}' not supported or not enabled"}
                
        except Exception as e:
            logger.error(f"Model export failed: {e}")
            return {"success": False, "error": str(e)}


# Global training engine instance
_training_engine = None


def get_training_engine() -> TrainingEngine:
    """Get the global training engine instance."""
    global _training_engine
    if _training_engine is None:
        _training_engine = TrainingEngine()
    return _training_engine


def initialize_training_engine() -> TrainingEngine:
    """Initialize the training engine."""
    settings = get_settings()
    
    if not settings.training_enabled:
        logger.info("Training pipeline disabled in configuration")
        return None
    
    engine = get_training_engine()
    logger.info("Training engine initialized successfully")
    return engine