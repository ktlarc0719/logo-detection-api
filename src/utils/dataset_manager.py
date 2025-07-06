"""
Dataset Management System for Logo Detection API

This module provides comprehensive dataset management capabilities including:
- Dataset creation and validation
- Image collection and annotation
- Dataset splitting and augmentation
- Sample dataset generation
- Dataset statistics and analysis
"""

import os
import json
import random
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import asyncio
import aiohttp
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetStats:
    """Dataset statistics container."""
    
    def __init__(self):
        self.total_images = 0
        self.total_labels = 0
        self.classes = {}
        self.split_distribution = {}
        self.image_sizes = []
        self.annotation_counts = []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_images": self.total_images,
            "total_labels": self.total_labels,
            "classes": self.classes,
            "split_distribution": self.split_distribution,
            "avg_image_size": {
                "width": np.mean([size[0] for size in self.image_sizes]) if self.image_sizes else 0,
                "height": np.mean([size[1] for size in self.image_sizes]) if self.image_sizes else 0
            },
            "avg_annotations_per_image": np.mean(self.annotation_counts) if self.annotation_counts else 0
        }


class DatasetManager:
    """Dataset management system."""
    
    def __init__(self):
        self.settings = get_settings()
        self.dataset_root = Path(self.settings.training_data_dir)
        self.logger = get_logger(__name__)
        
    def create_dataset_structure(self, dataset_name: str) -> Dict[str, Any]:
        """Create YOLO dataset directory structure."""
        try:
            dataset_path = self.dataset_root / dataset_name
            
            # Create directory structure
            directories = [
                dataset_path,
                dataset_path / "train" / "images",
                dataset_path / "train" / "labels",
                dataset_path / "val" / "images",
                dataset_path / "val" / "labels",
                dataset_path / "test" / "images",
                dataset_path / "test" / "labels"
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Create dataset.yaml file
            dataset_config = {
                "path": str(dataset_path.absolute()),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "nc": 0,  # Will be updated when classes are added
                "names": []  # Will be updated when classes are added
            }
            
            with open(dataset_path / "dataset.yaml", 'w') as f:
                yaml.dump(dataset_config, f)
            
            logger.info(f"Dataset structure created: {dataset_path}")
            
            return {
                "success": True,
                "dataset_path": str(dataset_path),
                "structure": [str(d.relative_to(dataset_path)) for d in directories[1:]]
            }
            
        except Exception as e:
            logger.error(f"Failed to create dataset structure: {e}")
            return {"success": False, "error": str(e)}
    
    def add_class_to_dataset(self, 
                           dataset_name: str, 
                           class_name: str) -> Dict[str, Any]:
        """Add a new class to existing dataset."""
        try:
            dataset_path = self.dataset_root / dataset_name
            dataset_yaml = dataset_path / "dataset.yaml"
            
            if not dataset_yaml.exists():
                return {"success": False, "error": "Dataset not found"}
            
            # Load existing config
            with open(dataset_yaml, 'r') as f:
                config = yaml.safe_load(f)
            
            # Add new class if not exists
            if class_name not in config["names"]:
                config["names"].append(class_name)
                config["nc"] = len(config["names"])
                
                # Save updated config
                with open(dataset_yaml, 'w') as f:
                    yaml.dump(config, f)
                
                logger.info(f"Added class '{class_name}' to dataset '{dataset_name}'")
                
                return {
                    "success": True,
                    "class_name": class_name,
                    "class_id": config["names"].index(class_name),
                    "total_classes": config["nc"]
                }
            else:
                return {
                    "success": False,
                    "error": f"Class '{class_name}' already exists in dataset"
                }
                
        except Exception as e:
            logger.error(f"Failed to add class to dataset: {e}")
            return {"success": False, "error": str(e)}
    
    def add_image_with_annotation(self,
                                 dataset_name: str,
                                 image_path: str,
                                 annotations: List[Dict[str, Any]],
                                 split: str = "train",
                                 preserve_filename: bool = False) -> Dict[str, Any]:
        """Add image with annotations to dataset."""
        try:
            dataset_path = self.dataset_root / dataset_name
            dataset_yaml = dataset_path / "dataset.yaml"
            
            if not dataset_yaml.exists():
                return {"success": False, "error": "Dataset not found"}
            
            # Load dataset config
            with open(dataset_yaml, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate split
            if split not in ["train", "val", "test"]:
                return {"success": False, "error": "Invalid split. Use 'train', 'val', or 'test'"}
            
            # Generate filename - preserve original if requested
            if preserve_filename:
                source_path = Path(image_path)
                image_name = source_path.stem
                image_ext = source_path.suffix
            else:
                # Generate unique filename
                image_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
                image_ext = Path(image_path).suffix
            
            # Copy image
            dest_image_path = dataset_path / split / "images" / f"{image_name}{image_ext}"
            shutil.copy2(image_path, dest_image_path)
            
            # Create YOLO format label file
            label_path = dataset_path / split / "labels" / f"{image_name}.txt"
            
            # Get image dimensions for normalization
            with Image.open(dest_image_path) as img:
                img_width, img_height = img.size
            
            # Convert annotations to YOLO format
            yolo_annotations = []
            for ann in annotations:
                class_name = ann["class_name"]
                bbox = ann["bbox"]  # Expected format: [x_min, y_min, x_max, y_max]
                
                # Get class ID
                if class_name not in config["names"]:
                    return {"success": False, "error": f"Class '{class_name}' not found in dataset"}
                
                class_id = config["names"].index(class_name)
                
                # Convert to YOLO format (normalized center x, center y, width, height)
                x_center = (bbox[0] + bbox[2]) / 2 / img_width
                y_center = (bbox[1] + bbox[3]) / 2 / img_height
                width = (bbox[2] - bbox[0]) / img_width
                height = (bbox[3] - bbox[1]) / img_height
                
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Save label file
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_annotations))
            
            logger.info(f"Added image with {len(annotations)} annotations to {split} split")
            
            return {
                "success": True,
                "image_path": str(dest_image_path),
                "label_path": str(label_path),
                "annotations_count": len(annotations),
                "split": split
            }
            
        except Exception as e:
            logger.error(f"Failed to add image with annotation: {e}")
            return {"success": False, "error": str(e)}
    
    def split_dataset(self,
                     dataset_name: str,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.2,
                     test_ratio: float = 0.1) -> Dict[str, Any]:
        """Split dataset into train/val/test sets."""
        try:
            # Validate ratios
            if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
                return {"success": False, "error": "Ratios must sum to 1.0"}
            
            dataset_path = self.dataset_root / dataset_name
            
            # Collect all images and labels
            all_images = []
            for split in ["train", "val", "test"]:
                images_dir = dataset_path / split / "images"
                if images_dir.exists():
                    for img_path in images_dir.glob("*"):
                        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                            label_path = dataset_path / split / "labels" / f"{img_path.stem}.txt"
                            if label_path.exists():
                                all_images.append((img_path, label_path, split))
            
            # Shuffle images
            random.shuffle(all_images)
            
            # Calculate split sizes
            total_images = len(all_images)
            train_size = int(total_images * train_ratio)
            val_size = int(total_images * val_ratio)
            test_size = total_images - train_size - val_size
            
            # Clear existing splits
            for split in ["train", "val", "test"]:
                for subdir in ["images", "labels"]:
                    split_dir = dataset_path / split / subdir
                    if split_dir.exists():
                        for file in split_dir.glob("*"):
                            file.unlink()
            
            # Redistribute images
            splits = [
                ("train", all_images[:train_size]),
                ("val", all_images[train_size:train_size + val_size]),
                ("test", all_images[train_size + val_size:])
            ]
            
            moved_counts = {}
            for split_name, images in splits:
                moved_counts[split_name] = 0
                for img_path, label_path, original_split in images:
                    # Move image
                    new_img_path = dataset_path / split_name / "images" / img_path.name
                    shutil.move(str(img_path), str(new_img_path))
                    
                    # Move label
                    new_label_path = dataset_path / split_name / "labels" / label_path.name
                    shutil.move(str(label_path), str(new_label_path))
                    
                    moved_counts[split_name] += 1
            
            logger.info(f"Dataset split completed: {moved_counts}")
            
            return {
                "success": True,
                "split_counts": moved_counts,
                "total_images": total_images,
                "ratios": {
                    "train": train_ratio,
                    "val": val_ratio,
                    "test": test_ratio
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to split dataset: {e}")
            return {"success": False, "error": str(e)}
    
    def get_dataset_statistics(self, dataset_name: str) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        try:
            dataset_path = self.dataset_root / dataset_name
            dataset_yaml = dataset_path / "dataset.yaml"
            
            if not dataset_yaml.exists():
                return {"success": False, "error": "Dataset not found"}
            
            # Load dataset config
            with open(dataset_yaml, 'r') as f:
                config = yaml.safe_load(f)
            
            stats = DatasetStats()
            
            # Analyze each split
            for split in ["train", "val", "test"]:
                images_dir = dataset_path / split / "images"
                labels_dir = dataset_path / split / "labels"
                
                if not images_dir.exists():
                    continue
                
                split_stats = {
                    "images": 0,
                    "labels": 0,
                    "classes": {},
                    "total_annotations": 0
                }
                
                # Count images and analyze labels
                for img_path in images_dir.glob("*"):
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        split_stats["images"] += 1
                        stats.total_images += 1
                        
                        # Get image size
                        try:
                            with Image.open(img_path) as img:
                                stats.image_sizes.append(img.size)
                        except Exception:
                            pass
                        
                        # Analyze corresponding label
                        label_path = labels_dir / f"{img_path.stem}.txt"
                        if label_path.exists():
                            split_stats["labels"] += 1
                            stats.total_labels += 1
                            
                            # Parse annotations
                            with open(label_path, 'r') as f:
                                annotations = f.read().strip().split('\n')
                                if annotations and annotations[0]:
                                    ann_count = len(annotations)
                                    split_stats["total_annotations"] += ann_count
                                    stats.annotation_counts.append(ann_count)
                                    
                                    for ann in annotations:
                                        parts = ann.split()
                                        if parts:
                                            class_id = int(parts[0])
                                            if class_id < len(config["names"]):
                                                class_name = config["names"][class_id]
                                                split_stats["classes"][class_name] = split_stats["classes"].get(class_name, 0) + 1
                                                stats.classes[class_name] = stats.classes.get(class_name, 0) + 1
                
                stats.split_distribution[split] = split_stats
            
            result = stats.to_dict()
            result.update({
                "success": True,
                "dataset_name": dataset_name,
                "dataset_path": str(dataset_path),
                "class_names": config["names"],
                "num_classes": config["nc"]
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get dataset statistics: {e}")
            return {"success": False, "error": str(e)}
    
    def validate_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Validate dataset for training."""
        try:
            dataset_path = self.dataset_root / dataset_name
            dataset_yaml = dataset_path / "dataset.yaml"
            
            if not dataset_yaml.exists():
                return {"success": False, "error": "Dataset not found"}
            
            # Load dataset config
            with open(dataset_yaml, 'r') as f:
                config = yaml.safe_load(f)
            
            validation_results = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "summary": {}
            }
            
            # Check minimum requirements
            stats = self.get_dataset_statistics(dataset_name)
            if not stats["success"]:
                validation_results["is_valid"] = False
                validation_results["errors"].append("Failed to get dataset statistics")
                return validation_results
            
            # Check if we have enough classes
            min_classes = 1
            if len(config["names"]) < min_classes:
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"Dataset needs at least {min_classes} classes")
            
            # Check if train split exists and has enough images
            train_images = stats["split_distribution"].get("train", {}).get("images", 0)
            min_train_images = self.settings.min_images_per_class * len(config["names"])
            
            if train_images < min_train_images:
                validation_results["is_valid"] = False
                validation_results["errors"].append(
                    f"Train split needs at least {min_train_images} images, found {train_images}"
                )
            
            # Check if validation split exists
            val_images = stats["split_distribution"].get("val", {}).get("images", 0)
            if val_images == 0:
                validation_results["warnings"].append("No validation split found")
            
            # Check class distribution
            for class_name in config["names"]:
                class_count = stats["classes"].get(class_name, 0)
                if class_count < self.settings.min_images_per_class:
                    validation_results["warnings"].append(
                        f"Class '{class_name}' has only {class_count} images, recommended minimum is {self.settings.min_images_per_class}"
                    )
            
            # Check for orphaned files
            orphaned_images = []
            orphaned_labels = []
            
            for split in ["train", "val", "test"]:
                images_dir = dataset_path / split / "images"
                labels_dir = dataset_path / split / "labels"
                
                if images_dir.exists():
                    for img_path in images_dir.glob("*"):
                        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                            label_path = labels_dir / f"{img_path.stem}.txt"
                            if not label_path.exists():
                                orphaned_images.append(str(img_path.relative_to(dataset_path)))
                
                if labels_dir.exists():
                    for label_path in labels_dir.glob("*.txt"):
                        img_found = False
                        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                            img_path = images_dir / f"{label_path.stem}{ext}"
                            if img_path.exists():
                                img_found = True
                                break
                        if not img_found:
                            orphaned_labels.append(str(label_path.relative_to(dataset_path)))
            
            if orphaned_images:
                validation_results["warnings"].append(f"Found {len(orphaned_images)} images without labels")
            
            if orphaned_labels:
                validation_results["warnings"].append(f"Found {len(orphaned_labels)} labels without images")
            
            validation_results["summary"] = {
                "total_images": stats["total_images"],
                "total_classes": len(config["names"]),
                "train_images": train_images,
                "val_images": val_images,
                "test_images": stats["split_distribution"].get("test", {}).get("images", 0),
                "orphaned_images": len(orphaned_images),
                "orphaned_labels": len(orphaned_labels)
            }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "is_valid": False,
                "errors": [str(e)]
            }
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets."""
        datasets = []
        
        if not self.dataset_root.exists():
            return datasets
        
        for dataset_dir in self.dataset_root.iterdir():
            if dataset_dir.is_dir():
                dataset_yaml = dataset_dir / "dataset.yaml"
                if dataset_yaml.exists():
                    try:
                        # Get basic info
                        stats = self.get_dataset_statistics(dataset_dir.name)
                        
                        dataset_info = {
                            "name": dataset_dir.name,
                            "path": str(dataset_dir),
                            "created": datetime.fromtimestamp(dataset_dir.stat().st_mtime).isoformat(),
                            "is_valid": False,
                            "total_images": 0,
                            "total_classes": 0
                        }
                        
                        if stats["success"]:
                            dataset_info.update({
                                "is_valid": True,
                                "total_images": stats["total_images"],
                                "total_classes": stats["num_classes"],
                                "class_names": stats["class_names"]
                            })
                        
                        datasets.append(dataset_info)
                        
                    except Exception as e:
                        logger.warning(f"Error reading dataset {dataset_dir.name}: {e}")
        
        return sorted(datasets, key=lambda x: x["created"], reverse=True)
    
    def delete_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Delete a dataset."""
        try:
            dataset_path = self.dataset_root / dataset_name
            
            if not dataset_path.exists():
                return {"success": False, "error": "Dataset not found"}
            
            # Remove dataset directory
            shutil.rmtree(dataset_path)
            
            logger.info(f"Dataset '{dataset_name}' deleted successfully")
            
            return {
                "success": True,
                "message": f"Dataset '{dataset_name}' deleted successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to delete dataset: {e}")
            return {"success": False, "error": str(e)}


# Global dataset manager instance
_dataset_manager = None


def get_dataset_manager() -> DatasetManager:
    """Get the global dataset manager instance."""
    global _dataset_manager
    if _dataset_manager is None:
        _dataset_manager = DatasetManager()
    return _dataset_manager