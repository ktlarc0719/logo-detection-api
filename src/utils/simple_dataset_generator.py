"""
Simple Dataset Generator for Logo Detection API

This module provides a simple and reliable way to generate synthetic datasets
without OpenCV dependencies, using only PIL for image generation.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import yaml

from src.core.config import get_settings
from src.utils.dataset_manager import get_dataset_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SimpleDatasetGenerator:
    """Simple dataset generator using only PIL."""
    
    def __init__(self):
        self.settings = get_settings()
        self.dataset_manager = get_dataset_manager()
        
        # Simple brand colors
        self.brand_colors = {
            "BANDAI": [(255, 0, 0), (255, 255, 255)],      # Red & White
            "Nintendo": [(220, 20, 60), (255, 255, 255)],   # Crimson & White
            "KONAMI": [(0, 100, 200), (255, 255, 255)],     # Blue & White
            "SONY": [(0, 0, 0), (255, 255, 255)],           # Black & White
            "Panasonic": [(0, 70, 140), (255, 255, 255)],   # Navy & White
        }
    
    def create_simple_logo_image(self, 
                                brand_name: str, 
                                width: int = 640, 
                                height: int = 480) -> Tuple[Image.Image, List[int]]:
        """Create a simple logo image with text."""
        
        # Create background with random color
        bg_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        image = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(image)
        
        # Get brand colors
        colors = self.brand_colors.get(brand_name, [(0, 0, 0), (255, 255, 255)])
        logo_bg_color, text_color = random.choice([colors, colors[::-1]])
        
        # Calculate font size based on image size and text length
        font_size = min(width, height) // (len(brand_name) + 2)
        font_size = max(24, min(80, font_size))
        
        # Try to use default font
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        
        # Calculate text size
        if font:
            bbox = draw.textbbox((0, 0), brand_name, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            # Approximate text size
            text_width = len(brand_name) * font_size * 0.6
            text_height = font_size
        
        # Center the logo
        logo_padding = 20
        logo_width = text_width + (logo_padding * 2)
        logo_height = text_height + (logo_padding * 2)
        
        # Position logo randomly but ensure it fits
        max_x = width - logo_width
        max_y = height - logo_height
        
        if max_x <= 0 or max_y <= 0:
            # Image too small, use smaller logo
            logo_x = logo_padding
            logo_y = logo_padding
            logo_width = width - (logo_padding * 2)
            logo_height = height - (logo_padding * 2)
        else:
            logo_x = random.randint(logo_padding, max_x)
            logo_y = random.randint(logo_padding, max_y)
        
        # Draw logo background rectangle
        rect_coords = [logo_x, logo_y, logo_x + logo_width, logo_y + logo_height]
        draw.rectangle(rect_coords, fill=logo_bg_color)
        
        # Add border
        border_color = tuple(max(0, c - 50) for c in logo_bg_color)
        draw.rectangle(rect_coords, outline=border_color, width=2)
        
        # Draw text centered in the rectangle
        text_x = logo_x + (logo_width - text_width) // 2
        text_y = logo_y + (logo_height - text_height) // 2
        
        if font:
            draw.text((text_x, text_y), brand_name, fill=text_color, font=font)
        else:
            # Fallback without font
            draw.text((text_x, text_y), brand_name, fill=text_color)
        
        # Return bounding box
        bbox = [logo_x, logo_y, logo_x + logo_width, logo_y + logo_height]
        
        return image, bbox
    
    def add_simple_variations(self, image: Image.Image) -> Image.Image:
        """Add simple variations to the image."""
        
        # Random brightness
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
        
        # Random contrast
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
        
        return image
    
    async def generate_simple_dataset(self,
                                    dataset_name: str,
                                    class_names: List[str],
                                    images_per_class: int = 20) -> Dict[str, Any]:
        """Generate a simple dataset with synthetic images."""
        try:
            logger.info(f"Generating simple dataset '{dataset_name}' with {len(class_names)} classes")
            
            # Ensure dataset exists
            datasets = self.dataset_manager.list_datasets()
            dataset_exists = any(d["name"] == dataset_name for d in datasets)
            
            if not dataset_exists:
                # Create dataset structure first
                create_result = self.dataset_manager.create_dataset_structure(dataset_name)
                if not create_result["success"]:
                    return create_result
                
                # Add classes to dataset
                for class_name in class_names:
                    class_result = self.dataset_manager.add_class_to_dataset(dataset_name, class_name)
                    if not class_result["success"]:
                        return class_result
            
            dataset_path = Path(self.settings.training_data_dir) / dataset_name
            
            # Create temp directory for images
            temp_dir = Path("temp") / "dataset_generation"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            total_images_generated = 0
            
            # Generate images for each class
            for class_name in class_names:
                logger.info(f"Generating {images_per_class} images for class '{class_name}'")
                
                for i in range(images_per_class):
                    try:
                        # Generate different image sizes
                        width = random.choice([512, 640, 800])
                        height = random.choice([384, 480, 600])
                        
                        # Create logo image
                        image, bbox = self.create_simple_logo_image(class_name, width, height)
                        
                        # Add variations
                        image = self.add_simple_variations(image)
                        
                        # Save temporary image
                        image_filename = f"{class_name}_{i:04d}_{random.randint(1000, 9999)}.jpg"
                        temp_image_path = temp_dir / image_filename
                        
                        # Save image
                        image.save(str(temp_image_path), "JPEG", quality=85)
                        
                        # Verify image was saved
                        if not temp_image_path.exists():
                            logger.error(f"Failed to save image: {temp_image_path}")
                            continue
                        
                        # Create annotation
                        annotations = [{
                            "class_name": class_name,
                            "bbox": bbox
                        }]
                        
                        # Randomly assign to train/val split (80/20)
                        split = "train" if random.random() < 0.8 else "val"
                        
                        # Add to dataset
                        add_result = self.dataset_manager.add_image_with_annotation(
                            dataset_name=dataset_name,
                            image_path=str(temp_image_path),
                            annotations=annotations,
                            split=split
                        )
                        
                        if add_result["success"]:
                            total_images_generated += 1
                            logger.debug(f"Successfully added image {image_filename} to {split} split")
                        else:
                            logger.warning(f"Failed to add image {image_filename}: {add_result.get('error', 'Unknown error')}")
                        
                        # Clean up temp file
                        if temp_image_path.exists():
                            temp_image_path.unlink()
                            
                    except Exception as e:
                        logger.error(f"Error generating image {i} for class {class_name}: {e}")
                        continue
            
            # Clean up temp directory
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Could not clean up temp directory: {e}")
            
            # Get final dataset statistics
            stats = self.dataset_manager.get_dataset_statistics(dataset_name)
            
            # Validate dataset
            validation = self.dataset_manager.validate_dataset(dataset_name)
            
            logger.info(f"Simple dataset generation completed: {total_images_generated} images generated")
            
            # Verify actual files were created
            train_images_dir = dataset_path / "train" / "images"
            actual_image_count = len(list(train_images_dir.glob("*.jpg"))) if train_images_dir.exists() else 0
            
            return {
                "success": True,
                "message": f"Simple dataset '{dataset_name}' generated successfully",
                "dataset_name": dataset_name,
                "total_images_generated": total_images_generated,
                "actual_images_created": actual_image_count,
                "classes": class_names,
                "images_per_class": images_per_class,
                "validation": validation,
                "statistics": stats if stats and stats.get("success") else None
            }
            
        except Exception as e:
            logger.error(f"Simple dataset generation failed: {e}")
            return {"success": False, "error": str(e)}


# Global simple dataset generator instance
_simple_generator = None


def get_simple_dataset_generator() -> SimpleDatasetGenerator:
    """Get the global simple dataset generator instance."""
    global _simple_generator
    if _simple_generator is None:
        _simple_generator = SimpleDatasetGenerator()
    return _simple_generator