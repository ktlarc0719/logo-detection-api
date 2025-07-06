"""
Minimal Dataset Generator for Logo Detection API

This module provides a minimal dataset generator with no external dependencies
except for PIL and standard library.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont


class MinimalDatasetGenerator:
    """Minimal dataset generator using only PIL and standard library."""
    
    def __init__(self, datasets_dir: str = "datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(exist_ok=True)
        
        # Simple brand colors
        self.brand_colors = {
            "BANDAI": [(255, 0, 0), (255, 255, 255)],      # Red & White
            "Nintendo": [(220, 20, 60), (255, 255, 255)],   # Crimson & White
            "KONAMI": [(0, 100, 200), (255, 255, 255)],     # Blue & White
            "SONY": [(0, 0, 0), (255, 255, 255)],           # Black & White
            "Panasonic": [(0, 70, 140), (255, 255, 255)],   # Navy & White
        }
    
    def create_dataset_structure(self, dataset_name: str, class_names: List[str]) -> bool:
        """Create YOLO dataset structure."""
        try:
            dataset_path = self.datasets_dir / dataset_name
            
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
            
            # Create simple dataset.yaml file
            dataset_config = f"""path: {dataset_path.absolute()}
train: train/images
val: val/images
test: test/images
nc: {len(class_names)}
names: {class_names}
"""
            
            with open(dataset_path / "dataset.yaml", 'w') as f:
                f.write(dataset_config)
            
            print(f"Created dataset structure: {dataset_path}")
            return True
            
        except Exception as e:
            print(f"Failed to create dataset structure: {e}")
            return False
    
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
        
        # Calculate font size
        font_size = min(width, height) // (len(brand_name) + 2)
        font_size = max(24, min(80, font_size))
        
        # Use default font
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        
        # Estimate text size (rough approximation)
        if font:
            # Use textbbox if available
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
        
        # Center the logo
        logo_padding = 20
        logo_width = int(text_width + (logo_padding * 2))
        logo_height = int(text_height + (logo_padding * 2))
        
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
            draw.text((int(text_x), int(text_y)), brand_name, fill=text_color, font=font)
        else:
            # Fallback without font
            draw.text((int(text_x), int(text_y)), brand_name, fill=text_color)
        
        # Return bounding box
        bbox = [logo_x, logo_y, logo_x + logo_width, logo_y + logo_height]
        
        return image, bbox
    
    def create_yolo_label(self, bbox: List[int], class_id: int, img_width: int, img_height: int) -> str:
        """Convert bounding box to YOLO format."""
        x_min, y_min, x_max, y_max = bbox
        
        # Convert to YOLO format (normalized center x, center y, width, height)
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    def generate_dataset(self,
                        dataset_name: str,
                        class_names: List[str],
                        images_per_class: int = 20) -> Dict[str, any]:
        """Generate a complete dataset with synthetic images."""
        try:
            print(f"Generating dataset '{dataset_name}' with {len(class_names)} classes")
            
            # Create dataset structure
            if not self.create_dataset_structure(dataset_name, class_names):
                return {"success": False, "error": "Failed to create dataset structure"}
            
            dataset_path = self.datasets_dir / dataset_name
            total_images_generated = 0
            
            # Generate images for each class
            for class_idx, class_name in enumerate(class_names):
                print(f"Generating {images_per_class} images for class '{class_name}'")
                
                for i in range(images_per_class):
                    try:
                        # Generate different image sizes
                        width = random.choice([512, 640, 800])
                        height = random.choice([384, 480, 600])
                        
                        # Create logo image
                        image, bbox = self.create_simple_logo_image(class_name, width, height)
                        
                        # Generate filename
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                        image_filename = f"{class_name}_{i:04d}_{timestamp}.jpg"
                        
                        # Randomly assign to train/val split (80/20)
                        split = "train" if random.random() < 0.8 else "val"
                        
                        # Save image
                        image_path = dataset_path / split / "images" / image_filename
                        image.save(str(image_path), "JPEG", quality=85)
                        
                        # Create label file
                        label_filename = image_filename.replace('.jpg', '.txt')
                        label_path = dataset_path / split / "labels" / label_filename
                        
                        yolo_label = self.create_yolo_label(bbox, class_idx, width, height)
                        
                        with open(label_path, 'w') as f:
                            f.write(yolo_label)
                        
                        total_images_generated += 1
                        
                        if i % 10 == 0:
                            print(f"  Generated {i+1}/{images_per_class} images for {class_name}")
                            
                    except Exception as e:
                        print(f"Error generating image {i} for class {class_name}: {e}")
                        continue
            
            # Count actual files created
            train_images_dir = dataset_path / "train" / "images"
            val_images_dir = dataset_path / "val" / "images"
            
            actual_train_images = len(list(train_images_dir.glob("*.jpg"))) if train_images_dir.exists() else 0
            actual_val_images = len(list(val_images_dir.glob("*.jpg"))) if val_images_dir.exists() else 0
            actual_total = actual_train_images + actual_val_images
            
            print(f"Dataset generation completed!")
            print(f"Generated: {total_images_generated} images")
            print(f"Actual files: {actual_total} images ({actual_train_images} train, {actual_val_images} val)")
            
            return {
                "success": True,
                "dataset_name": dataset_name,
                "total_images_generated": total_images_generated,
                "actual_images_created": actual_total,
                "train_images": actual_train_images,
                "val_images": actual_val_images,
                "classes": class_names,
                "images_per_class": images_per_class,
                "dataset_path": str(dataset_path)
            }
            
        except Exception as e:
            print(f"Dataset generation failed: {e}")
            return {"success": False, "error": str(e)}


def test_generator():
    """Test the minimal dataset generator."""
    generator = MinimalDatasetGenerator()
    
    result = generator.generate_dataset(
        dataset_name="test_minimal_dataset",
        class_names=["BANDAI", "Nintendo", "SONY"],
        images_per_class=5
    )
    
    print(f"Test result: {result}")
    return result


if __name__ == "__main__":
    test_generator()