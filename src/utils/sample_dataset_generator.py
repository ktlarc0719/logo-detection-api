"""
Sample Dataset Generator for Logo Detection API

This module generates synthetic datasets for testing and demonstration purposes.
It creates placeholder images with text logos and proper YOLO annotations.
"""

import os
import random
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import yaml

# Check if OpenCV is available, use fallback if not
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available, using PIL-only image generation")

from src.core.config import get_settings
from src.utils.dataset_manager import get_dataset_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SampleDatasetGenerator:
    """Generate sample datasets with synthetic logo images."""
    
    def __init__(self):
        self.settings = get_settings()
        self.dataset_manager = get_dataset_manager()
        
        # Default colors for different brands
        self.brand_colors = {
            "BANDAI": [(255, 0, 0), (255, 255, 255)],      # Red & White
            "Nintendo": [(220, 20, 60), (255, 255, 255)],   # Crimson & White
            "KONAMI": [(0, 100, 200), (255, 255, 255)],     # Blue & White
            "SONY": [(0, 0, 0), (255, 255, 255)],           # Black & White
            "Panasonic": [(0, 70, 140), (255, 255, 255)],   # Navy & White
            "Samsung": [(30, 144, 255), (255, 255, 255)],   # DodgerBlue & White
            "LG": [(200, 20, 60), (255, 255, 255)],         # Crimson & White
            "Canon": [(220, 20, 60), (255, 255, 255)],      # Red & White
            "Nikon": [(255, 215, 0), (0, 0, 0)],            # Gold & Black
            "Apple": [(128, 128, 128), (255, 255, 255)]     # Gray & White
        }
        
        # Background patterns
        self.background_types = [
            "solid",
            "gradient",
            "noise",
            "textured"
        ]
    
    def _get_random_font_size(self, text_length: int, image_width: int) -> int:
        """Calculate appropriate font size based on text length and image width."""
        base_size = image_width // (text_length + 2)
        return max(20, min(80, base_size))
    
    def _create_background(self, width: int, height: int, bg_type: str = "solid") -> Image.Image:
        """Create different types of backgrounds using PIL."""
        if bg_type == "solid":
            # Random solid color
            color = tuple(random.randint(200, 255) for _ in range(3))
            background = Image.new('RGB', (width, height), color)
            
        elif bg_type == "gradient":
            # Gradient background
            background = Image.new('RGB', (width, height))
            start_color = tuple(random.randint(180, 255) for _ in range(3))
            end_color = tuple(random.randint(180, 255) for _ in range(3))
            
            pixels = []
            for y in range(height):
                ratio = y / height
                color = tuple(int(start_color[i] * (1 - ratio) + end_color[i] * ratio) for i in range(3))
                for x in range(width):
                    pixels.append(color)
            
            background.putdata(pixels)
                
        elif bg_type == "noise":
            # Noisy background
            pixels = []
            for _ in range(width * height):
                color = tuple(random.randint(220, 255) for _ in range(3))
                pixels.append(color)
            background = Image.new('RGB', (width, height))
            background.putdata(pixels)
            
        elif bg_type == "textured":
            # Simple textured background
            background = Image.new('RGB', (width, height), (240, 240, 240))
            # Add some texture using filter
            background = background.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        else:
            # Default solid white
            background = Image.new('RGB', (width, height), (255, 255, 255))
        
        return background
    
    def _add_random_elements(self, image: Image.Image) -> Image.Image:
        """Add random elements to make the image more realistic using PIL."""
        width, height = image.size
        
        # Add random shapes occasionally
        if random.random() < 0.3:
            overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            for _ in range(random.randint(1, 3)):
                if random.random() < 0.5:
                    # Rectangle
                    x1, y1 = random.randint(0, width//2), random.randint(0, height//2)
                    x2, y2 = random.randint(width//2, width), random.randint(height//2, height)
                    color = tuple(random.randint(0, 100) for _ in range(3)) + (30,)  # Add alpha
                    draw.rectangle([x1, y1, x2, y2], fill=color)
                else:
                    # Circle
                    center_x = random.randint(0, width)
                    center_y = random.randint(0, height)
                    radius = random.randint(10, min(width, height) // 8)
                    color = tuple(random.randint(0, 100) for _ in range(3)) + (30,)  # Add alpha
                    draw.ellipse([center_x-radius, center_y-radius, center_x+radius, center_y+radius], fill=color)
            
            # Composite overlay on original
            image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        
        return image
    
    def _create_logo_image(self, 
                          brand_name: str, 
                          width: int = 640, 
                          height: int = 480) -> Tuple[Image.Image, List[int]]:
        """Create a synthetic logo image with bounding box."""
        
        # Create background
        bg_type = random.choice(self.background_types)
        pil_image = self._create_background(width, height, bg_type)
        
        # Get brand colors
        colors = self.brand_colors.get(brand_name, [(0, 0, 0), (255, 255, 255)])
        bg_color, text_color = random.choice([colors, colors[::-1]])
        
        # Get draw object
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a system font, fallback to default
        try:
            font_size = self._get_random_font_size(len(brand_name), width)
            # Try common system fonts
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/System/Library/Fonts/Arial.ttf",  # macOS
                "C:/Windows/Fonts/arial.ttf"        # Windows
            ]
            
            font = None
            for font_path in font_paths:
                try:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, font_size)
                        break
                except Exception:
                    continue
            
            if font is None:
                font = ImageFont.load_default()
                
        except Exception:
            font = ImageFont.load_default()
        
        # Calculate text position
        try:
            bbox = draw.textbbox((0, 0), brand_name, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except Exception:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(brand_name, font=font)
        
        # Random position with some padding
        padding = 50
        max_x = max(padding, width - text_width - padding)
        max_y = max(padding, height - text_height - padding)
        
        x = random.randint(padding, max_x)
        y = random.randint(padding, max_y)
        
        # Add background rectangle for logo
        logo_padding = 20
        rect_x1 = max(0, x - logo_padding)
        rect_y1 = max(0, y - logo_padding)
        rect_x2 = min(width, x + text_width + logo_padding)
        rect_y2 = min(height, y + text_height + logo_padding)
        
        # Draw background rectangle
        draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill=bg_color, outline=None)
        
        # Add border occasionally
        if random.random() < 0.4:
            border_color = tuple(max(0, c - 50) for c in bg_color)
            draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill=None, outline=border_color, width=2)
        
        # Draw text
        draw.text((x, y), brand_name, fill=text_color, font=font)
        
        # Add random elements
        pil_image = self._add_random_elements(pil_image)
        
        # Return bounding box in [x_min, y_min, x_max, y_max] format
        bbox = [rect_x1, rect_y1, rect_x2, rect_y2]
        
        return pil_image, bbox
    
    def _add_noise_and_variations(self, image: Image.Image) -> Image.Image:
        """Add realistic variations to the image using PIL."""
        # Random brightness/contrast adjustment
        if random.random() < 0.6:
            # Adjust brightness
            if random.random() < 0.5:
                enhancer = ImageEnhance.Brightness(image)
                factor = random.uniform(0.8, 1.2)
                image = enhancer.enhance(factor)
            
            # Adjust contrast
            if random.random() < 0.5:
                enhancer = ImageEnhance.Contrast(image)
                factor = random.uniform(0.8, 1.2)
                image = enhancer.enhance(factor)
        
        # Random blur
        if random.random() < 0.3:
            radius = random.uniform(0.5, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        # Random color adjustment
        if random.random() < 0.4:
            enhancer = ImageEnhance.Color(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
        
        return image
    
    async def generate_sample_dataset(self,
                                    dataset_name: str,
                                    class_names: List[str],
                                    images_per_class: int = 20) -> Dict[str, Any]:
        """Generate a complete sample dataset."""
        try:
            logger.info(f"Generating sample dataset '{dataset_name}' with {len(class_names)} classes")
            
            # Create dataset structure
            result = self.dataset_manager.create_dataset_structure(dataset_name)
            if not result["success"]:
                return result
            
            # Add classes to dataset
            for class_name in class_names:
                class_result = self.dataset_manager.add_class_to_dataset(dataset_name, class_name)
                if not class_result["success"]:
                    return class_result
            
            dataset_path = Path(self.settings.training_data_dir) / dataset_name
            temp_images_dir = dataset_path / "temp_images"
            temp_images_dir.mkdir(exist_ok=True)
            
            total_images_generated = 0
            
            # Generate images for each class
            for class_name in class_names:
                logger.info(f"Generating {images_per_class} images for class '{class_name}'")
                
                for i in range(images_per_class):
                    # Generate different image sizes
                    width = random.choice([640, 800, 1024, 512])
                    height = random.choice([480, 600, 768, 384])
                    
                    # Create logo image
                    image, bbox = self._create_logo_image(class_name, width, height)
                    
                    # Add variations
                    image = self._add_noise_and_variations(image)
                    
                    # Save temporary image
                    image_filename = f"{class_name}_{i:04d}.jpg"
                    temp_image_path = temp_images_dir / image_filename
                    
                    # Save using PIL
                    image.save(str(temp_image_path), "JPEG", quality=90)
                    
                    # Add to dataset with annotation
                    annotations = [{
                        "class_name": class_name,
                        "bbox": bbox
                    }]
                    
                    # Randomly assign to train/val split (80/20)
                    split = "train" if random.random() < 0.8 else "val"
                    
                    add_result = self.dataset_manager.add_image_with_annotation(
                        dataset_name=dataset_name,
                        image_path=str(temp_image_path),
                        annotations=annotations,
                        split=split
                    )
                    
                    if add_result["success"]:
                        total_images_generated += 1
                    else:
                        logger.warning(f"Failed to add image {image_filename}: {add_result['error']}")
                    
                    # Clean up temporary file
                    if temp_image_path.exists():
                        temp_image_path.unlink()
            
            # Clean up temp directory
            if temp_images_dir.exists():
                temp_images_dir.rmdir()
            
            # Get final dataset statistics
            stats = self.dataset_manager.get_dataset_statistics(dataset_name)
            
            # Validate dataset
            validation = self.dataset_manager.validate_dataset(dataset_name)
            
            logger.info(f"Sample dataset generation completed: {total_images_generated} images generated")
            
            return {
                "success": True,
                "message": f"Sample dataset '{dataset_name}' generated successfully",
                "dataset_name": dataset_name,
                "total_images_generated": total_images_generated,
                "classes": class_names,
                "images_per_class": images_per_class,
                "validation": validation,
                "statistics": stats if stats["success"] else None
            }
            
        except Exception as e:
            logger.error(f"Sample dataset generation failed: {e}")
            
            # Clean up on failure
            try:
                self.dataset_manager.delete_dataset(dataset_name)
            except Exception:
                pass
            
            return {"success": False, "error": str(e)}
    
    async def generate_quick_demo_dataset(self) -> Dict[str, Any]:
        """Generate a quick demo dataset with default classes."""
        demo_classes = self.settings.initial_logo_classes
        
        return await self.generate_sample_dataset(
            dataset_name="demo_dataset",
            class_names=demo_classes,
            images_per_class=15
        )
    
    def create_augmented_variants(self, 
                                 image: Image.Image, 
                                 bbox: List[int], 
                                 num_variants: int = 5) -> List[Tuple[Image.Image, List[int]]]:
        """Create augmented variants of an image while preserving bounding box."""
        variants = []
        width, height = image.size
        
        for _ in range(num_variants):
            variant_image = image.copy()
            variant_bbox = bbox.copy()
            
            # Random rotation (small angles)
            if random.random() < 0.3:
                angle = random.uniform(-10, 10)
                variant_image = variant_image.rotate(angle, expand=False, fillcolor=(255, 255, 255))
                # Note: bbox adjustment for rotation is complex, skipping for simplicity
            
            # Random scaling
            if random.random() < 0.4:
                scale = random.uniform(0.9, 1.1)
                new_width = int(width * scale)
                new_height = int(height * scale)
                variant_image = variant_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Adjust bbox
                variant_bbox = [int(coord * scale) for coord in bbox]
                
                # Crop or pad to original size
                if scale > 1.0:
                    # Crop to original size
                    x_offset = (new_width - width) // 2
                    y_offset = (new_height - height) // 2
                    variant_image = variant_image.crop((x_offset, y_offset, x_offset + width, y_offset + height))
                    variant_bbox[0] -= x_offset
                    variant_bbox[1] -= y_offset
                    variant_bbox[2] -= x_offset
                    variant_bbox[3] -= y_offset
                else:
                    # Pad to original size
                    pad_x = (width - new_width) // 2
                    pad_y = (height - new_height) // 2
                    new_img = Image.new('RGB', (width, height), (128, 128, 128))
                    new_img.paste(variant_image, (pad_x, pad_y))
                    variant_image = new_img
                    variant_bbox[0] += pad_x
                    variant_bbox[1] += pad_y
                    variant_bbox[2] += pad_x
                    variant_bbox[3] += pad_y
            
            # Add noise and variations
            variant_image = self._add_noise_and_variations(variant_image)
            
            # Ensure bbox is within image bounds
            variant_bbox[0] = max(0, variant_bbox[0])
            variant_bbox[1] = max(0, variant_bbox[1])
            variant_bbox[2] = min(width, variant_bbox[2])
            variant_bbox[3] = min(height, variant_bbox[3])
            
            variants.append((variant_image, variant_bbox))
        
        return variants