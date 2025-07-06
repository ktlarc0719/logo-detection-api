"""
Robust Dataset Generator for Logo Detection API

This module provides a robust and reliable dataset generator that ensures
images and labels are correctly created and can be properly detected by the API.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import yaml

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RobustDatasetGenerator:
    """Robust dataset generator with improved reliability."""
    
    def __init__(self):
        self.settings = get_settings()
        self.datasets_dir = Path(self.settings.training_data_dir)
        self.datasets_dir.mkdir(exist_ok=True)
        
        # Enhanced brand colors with more variety
        self.brand_colors = {
            "BANDAI": [
                [(255, 0, 0), (255, 255, 255)],      # Red & White
                [(255, 51, 51), (0, 0, 0)],          # Light Red & Black
                [(204, 0, 0), (255, 255, 255)]       # Dark Red & White
            ],
            "Nintendo": [
                [(220, 20, 60), (255, 255, 255)],    # Crimson & White
                [(255, 255, 255), (220, 20, 60)],    # White & Crimson
                [(0, 0, 0), (255, 255, 255)]         # Black & White
            ],
            "KONAMI": [
                [(0, 100, 200), (255, 255, 255)],    # Blue & White
                [(255, 255, 255), (0, 100, 200)],    # White & Blue
                [(0, 0, 0), (255, 255, 255)]         # Black & White
            ],
            "SONY": [
                [(0, 0, 0), (255, 255, 255)],        # Black & White
                [(255, 255, 255), (0, 0, 0)],        # White & Black
                [(50, 50, 50), (255, 255, 255)]      # Dark Gray & White
            ],
            "Panasonic": [
                [(0, 70, 140), (255, 255, 255)],     # Navy & White
                [(255, 255, 255), (0, 70, 140)],     # White & Navy
                [(0, 0, 0), (255, 255, 255)]         # Black & White
            ]
        }
        
        # Background types
        self.background_types = ["solid", "gradient", "textured"]
    
    def create_dataset_structure(self, dataset_name: str, class_names: List[str]) -> Dict[str, Any]:
        """Create YOLO dataset structure with proper validation."""
        try:
            dataset_path = self.datasets_dir / dataset_name
            
            # Remove existing dataset if it exists
            if dataset_path.exists():
                shutil.rmtree(dataset_path)
                logger.info(f"Removed existing dataset: {dataset_path}")
            
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
                logger.debug(f"Created directory: {directory}")
            
            # Create dataset.yaml file with proper YOLO format
            dataset_config = {
                "path": str(dataset_path.absolute()),
                "train": "train/images",
                "val": "val/images", 
                "test": "test/images",
                "nc": len(class_names),
                "names": class_names
            }
            
            yaml_path = dataset_path / "dataset.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(dataset_config, f, default_flow_style=False)
            
            logger.info(f"Created dataset structure: {dataset_path}")
            
            return {
                "success": True,
                "dataset_path": str(dataset_path),
                "yaml_path": str(yaml_path),
                "directories": [str(d) for d in directories]
            }
            
        except Exception as e:
            logger.error(f"Failed to create dataset structure: {e}")
            return {"success": False, "error": str(e)}
    
    def create_background(self, width: int, height: int, bg_type: str = "solid") -> Image.Image:
        """Create varied background images."""
        if bg_type == "solid":
            # Random solid color backgrounds
            bg_colors = [
                (255, 255, 255),  # White
                (240, 240, 240),  # Light gray
                (220, 220, 220),  # Gray
                (250, 250, 250),  # Off-white
                (245, 245, 245)   # Very light gray
            ]
            color = random.choice(bg_colors)
            return Image.new('RGB', (width, height), color)
            
        elif bg_type == "gradient":
            # Simple gradient background
            image = Image.new('RGB', (width, height))
            start_color = random.choice([(255, 255, 255), (240, 240, 240), (250, 250, 250)])
            end_color = random.choice([(220, 220, 220), (200, 200, 200), (230, 230, 230)])
            
            pixels = []
            for y in range(height):
                ratio = y / height
                color = tuple(int(start_color[i] * (1 - ratio) + end_color[i] * ratio) for i in range(3))
                for x in range(width):
                    pixels.append(color)
            
            image.putdata(pixels)
            return image
            
        elif bg_type == "textured":
            # Textured background with subtle noise
            base_color = random.choice([(245, 245, 245), (240, 240, 240), (250, 250, 250)])
            image = Image.new('RGB', (width, height), base_color)
            
            # Add subtle texture
            pixels = list(image.getdata())
            noisy_pixels = []
            
            for pixel in pixels:
                noise = random.randint(-5, 5)
                new_pixel = tuple(max(0, min(255, c + noise)) for c in pixel)
                noisy_pixels.append(new_pixel)
            
            image.putdata(noisy_pixels)
            return image
        
        else:
            # Default white background
            return Image.new('RGB', (width, height), (255, 255, 255))
    
    def create_logo_image(self, brand_name: str, width: int = 640, height: int = 480) -> Tuple[Image.Image, List[int]]:
        """Create a realistic logo image with proper bounding box."""
        
        # Create background
        bg_type = random.choice(self.background_types)
        image = self.create_background(width, height, bg_type)
        draw = ImageDraw.Draw(image)
        
        # Get brand colors
        color_sets = self.brand_colors.get(brand_name, [[(0, 0, 0), (255, 255, 255)]])
        bg_color, text_color = random.choice(color_sets)
        
        # Calculate appropriate font size
        base_font_size = min(width, height) // 12
        font_size = random.randint(max(20, base_font_size - 10), base_font_size + 10)
        
        # Load font
        font = None
        try:
            font = ImageFont.load_default()
        except Exception:
            pass
        
        # Get text dimensions
        if font:
            try:
                bbox = draw.textbbox((0, 0), brand_name, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                # Fallback for older PIL versions
                text_width = len(brand_name) * font_size * 0.6
                text_height = font_size
        else:
            text_width = len(brand_name) * font_size * 0.6
            text_height = font_size
        
        # Calculate logo dimensions with padding
        padding = random.randint(15, 30)
        logo_width = int(text_width + padding * 2)
        logo_height = int(text_height + padding * 2)
        
        # Position logo (ensure it fits within image)
        margin = 20
        max_x = max(margin, width - logo_width - margin)
        max_y = max(margin, height - logo_height - margin)
        
        if max_x <= margin:
            logo_x = margin
            logo_width = width - margin * 2
        else:
            logo_x = random.randint(margin, max_x)
        
        if max_y <= margin:
            logo_y = margin
            logo_height = height - margin * 2
        else:
            logo_y = random.randint(margin, max_y)
        
        # Draw logo background
        logo_rect = [logo_x, logo_y, logo_x + logo_width, logo_y + logo_height]
        
        # Add some shape variety
        if random.random() < 0.3:
            # Rounded rectangle effect
            draw.rectangle(logo_rect, fill=bg_color)
        else:
            # Regular rectangle
            draw.rectangle(logo_rect, fill=bg_color)
        
        # Add border occasionally
        if random.random() < 0.4:
            border_width = random.randint(1, 3)
            border_color = tuple(max(0, c - 30) for c in bg_color)
            draw.rectangle(logo_rect, outline=border_color, width=border_width)
        
        # Calculate text position (centered in logo box)
        text_x = logo_x + (logo_width - text_width) // 2
        text_y = logo_y + (logo_height - text_height) // 2
        
        # Draw text
        if font:
            draw.text((int(text_x), int(text_y)), brand_name, fill=text_color, font=font)
        else:
            draw.text((int(text_x), int(text_y)), brand_name, fill=text_color)
        
        # Add visual enhancements occasionally
        if random.random() < 0.2:
            # Add shadow effect
            shadow_offset = 2
            shadow_color = tuple(max(0, c - 50) for c in bg_color)
            if font:
                draw.text((int(text_x + shadow_offset), int(text_y + shadow_offset)), 
                         brand_name, fill=shadow_color, font=font)
            else:
                draw.text((int(text_x + shadow_offset), int(text_y + shadow_offset)), 
                         brand_name, fill=shadow_color)
            
            # Redraw main text on top
            if font:
                draw.text((int(text_x), int(text_y)), brand_name, fill=text_color, font=font)
            else:
                draw.text((int(text_x), int(text_y)), brand_name, fill=text_color)
        
        # Apply image enhancements
        if random.random() < 0.3:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.9, 1.1))
        
        if random.random() < 0.3:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.9, 1.1))
        
        # Return bounding box [x_min, y_min, x_max, y_max]
        bbox = [logo_x, logo_y, logo_x + logo_width, logo_y + logo_height]
        
        return image, bbox
    
    def create_yolo_label(self, bbox: List[int], class_id: int, img_width: int, img_height: int) -> str:
        """Create YOLO format label string."""
        x_min, y_min, x_max, y_max = bbox
        
        # Convert to YOLO format (normalized center x, center y, width, height)
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # Ensure values are within valid range [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    def generate_robust_dataset(self,
                               dataset_name: str,
                               class_names: List[str],
                               images_per_class: int = 30) -> Dict[str, Any]:
        """Generate a robust dataset with comprehensive validation."""
        try:
            logger.info(f"Starting robust dataset generation: {dataset_name}")
            logger.info(f"Classes: {class_names}, Images per class: {images_per_class}")
            
            # Step 1: Create dataset structure
            structure_result = self.create_dataset_structure(dataset_name, class_names)
            if not structure_result["success"]:
                return structure_result
            
            dataset_path = Path(structure_result["dataset_path"])
            total_generated = 0
            generation_stats = {
                "train": {"images": 0, "labels": 0},
                "val": {"images": 0, "labels": 0},
                "test": {"images": 0, "labels": 0}
            }
            
            # Step 2: Generate images for each class
            for class_idx, class_name in enumerate(class_names):
                logger.info(f"Generating images for class '{class_name}' (ID: {class_idx})")
                
                class_generated = 0
                
                for i in range(images_per_class):
                    try:
                        # Generate varied image sizes
                        sizes = [(512, 384), (640, 480), (800, 600), (768, 576)]
                        width, height = random.choice(sizes)
                        
                        # Create logo image
                        image, bbox = self.create_logo_image(class_name, width, height)
                        
                        # Generate unique filename
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                        image_filename = f"{class_name}_{i:04d}_{timestamp}.jpg"
                        
                        # Determine split (80% train, 15% val, 5% test)
                        rand = random.random()
                        if rand < 0.80:
                            split = "train"
                        elif rand < 0.95:
                            split = "val"
                        else:
                            split = "test"
                        
                        # Save image
                        image_path = dataset_path / split / "images" / image_filename
                        image.save(str(image_path), "JPEG", quality=random.randint(85, 95), optimize=True)
                        
                        # Verify image was saved
                        if not image_path.exists() or image_path.stat().st_size == 0:
                            logger.error(f"Failed to save image: {image_path}")
                            continue
                        
                        # Create and save label
                        label_filename = image_filename.replace('.jpg', '.txt')
                        label_path = dataset_path / split / "labels" / label_filename
                        
                        yolo_label = self.create_yolo_label(bbox, class_idx, width, height)
                        
                        with open(label_path, 'w') as f:
                            f.write(yolo_label + '\n')
                        
                        # Verify label was saved
                        if not label_path.exists() or label_path.stat().st_size == 0:
                            logger.error(f"Failed to save label: {label_path}")
                            # Remove orphaned image
                            if image_path.exists():
                                image_path.unlink()
                            continue
                        
                        # Update counters
                        generation_stats[split]["images"] += 1
                        generation_stats[split]["labels"] += 1
                        total_generated += 1
                        class_generated += 1
                        
                        if (i + 1) % 10 == 0:
                            logger.info(f"  Generated {i + 1}/{images_per_class} images for {class_name}")
                        
                    except Exception as e:
                        logger.error(f"Error generating image {i} for class {class_name}: {e}")
                        continue
                
                logger.info(f"Completed class '{class_name}': {class_generated} images generated")
            
            # Step 3: Validate generated dataset
            validation_result = self.validate_generated_dataset(dataset_path, class_names)
            
            # Step 4: Create summary
            result = {
                "success": True,
                "dataset_name": dataset_name,
                "dataset_path": str(dataset_path),
                "total_images_generated": total_generated,
                "generation_stats": generation_stats,
                "classes": class_names,
                "num_classes": len(class_names),
                "images_per_class": images_per_class,
                "validation": validation_result,
                "ready_for_training": validation_result["is_valid"]
            }
            
            logger.info(f"Dataset generation completed successfully!")
            logger.info(f"Total images: {total_generated}")
            logger.info(f"Train: {generation_stats['train']['images']}, Val: {generation_stats['val']['images']}, Test: {generation_stats['test']['images']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Robust dataset generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def validate_generated_dataset(self, dataset_path: Path, class_names: List[str]) -> Dict[str, Any]:
        """Thoroughly validate the generated dataset."""
        try:
            validation = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "file_counts": {},
                "class_distribution": {}
            }
            
            # Check each split
            for split in ["train", "val", "test"]:
                images_dir = dataset_path / split / "images"
                labels_dir = dataset_path / split / "labels"
                
                if not images_dir.exists():
                    validation["errors"].append(f"Missing {split}/images directory")
                    validation["is_valid"] = False
                    continue
                
                if not labels_dir.exists():
                    validation["errors"].append(f"Missing {split}/labels directory")
                    validation["is_valid"] = False
                    continue
                
                # Count files
                image_files = list(images_dir.glob("*.jpg"))
                label_files = list(labels_dir.glob("*.txt"))
                
                validation["file_counts"][split] = {
                    "images": len(image_files),
                    "labels": len(label_files)
                }
                
                # Check for matching image-label pairs
                image_stems = {f.stem for f in image_files}
                label_stems = {f.stem for f in label_files}
                
                orphaned_images = image_stems - label_stems
                orphaned_labels = label_stems - image_stems
                
                if orphaned_images:
                    validation["warnings"].append(f"{split}: {len(orphaned_images)} images without labels")
                
                if orphaned_labels:
                    validation["warnings"].append(f"{split}: {len(orphaned_labels)} labels without images")
                
                # Analyze class distribution
                class_counts = {name: 0 for name in class_names}
                
                for label_file in label_files:
                    try:
                        with open(label_file, 'r') as f:
                            content = f.read().strip()
                            if content:
                                class_id = int(content.split()[0])
                                if 0 <= class_id < len(class_names):
                                    class_counts[class_names[class_id]] += 1
                    except Exception as e:
                        validation["warnings"].append(f"Error reading label {label_file}: {e}")
                
                validation["class_distribution"][split] = class_counts
            
            # Check minimum requirements for training
            train_images = validation["file_counts"].get("train", {}).get("images", 0)
            min_required = len(class_names) * 10  # At least 10 images per class
            
            if train_images < min_required:
                validation["errors"].append(f"Insufficient training images: {train_images} < {min_required}")
                validation["is_valid"] = False
            
            # Check dataset.yaml
            yaml_path = dataset_path / "dataset.yaml"
            if not yaml_path.exists():
                validation["errors"].append("Missing dataset.yaml file")
                validation["is_valid"] = False
            else:
                try:
                    with open(yaml_path, 'r') as f:
                        yaml_content = yaml.safe_load(f)
                        if yaml_content.get("nc") != len(class_names):
                            validation["warnings"].append("Class count mismatch in dataset.yaml")
                except Exception as e:
                    validation["errors"].append(f"Invalid dataset.yaml: {e}")
            
            return validation
            
        except Exception as e:
            return {
                "is_valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
                "file_counts": {},
                "class_distribution": {}
            }


def get_robust_dataset_generator() -> RobustDatasetGenerator:
    """Get robust dataset generator instance."""
    return RobustDatasetGenerator()