"""
Real Logo Adapter for Dataset Generation API

This module adapts the RealLogoDatasetGenerator to work with the API's
generate-sample endpoint, replacing the text-based generation with real logo images.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
import shutil

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from real_logo_dataset_generator import RealLogoDatasetGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RealLogoAdapter:
    """Adapter to use real logo images in the dataset generation API."""
    
    def __init__(self):
        self.generator = RealLogoDatasetGenerator()
        
        # ロゴファイルのマッピング（実際に存在するファイルのみ）
        self.logo_mappings = {
            "BANDAI": "data/logos/input/BANDAI/bandai_logo1.png",
            "Apple": "data/logos/input/Apple_logo.jpg",
            "Google": "data/logos/input/Google_logo.jpg",
            "Microsoft": "data/logos/input/Microsoft_logo.jpg",
            "Amazon": "data/logos/input/Amazon_logo.jpg",
            "Meta": "data/logos/input/Meta_logo.jpg"
        }
    
    def generate_robust_dataset(self, 
                              dataset_name: str,
                              class_names: List[str],
                              images_per_class: int = 20) -> Dict[str, Any]:
        """Generate dataset using real logo images."""
        
        try:
            logger.info(f"Generating real logo dataset: {dataset_name}")
            logger.info(f"Classes: {class_names}, Images per class: {images_per_class}")
            
            # Validate all requested classes have logo mappings
            missing_logos = []
            for class_name in class_names:
                if class_name not in self.logo_mappings:
                    missing_logos.append(class_name)
            
            if missing_logos:
                return {
                    "success": False,
                    "error": f"No logo mapping found for classes: {missing_logos}. Available classes: {list(self.logo_mappings.keys())}"
                }
            
            # Check if all logo files exist
            missing_files = []
            for class_name in class_names:
                logo_path = Path(self.logo_mappings[class_name])
                if not logo_path.exists():
                    missing_files.append(f"{class_name}: {logo_path}")
            
            if missing_files:
                return {
                    "success": False,
                    "error": f"Logo files not found: {missing_files}"
                }
            
            # Generate dataset for first class (creates the dataset)
            first_class = class_names[0]
            success = self.generator.create_dataset_with_variations(
                dataset_name=dataset_name,
                logo_path=self.logo_mappings[first_class],
                brand_name=first_class,
                num_variations=images_per_class
            )
            
            if not success:
                return {
                    "success": False,
                    "error": f"Failed to generate dataset for {first_class}"
                }
            
            # Generate for remaining classes (adds to existing dataset)
            for class_name in class_names[1:]:
                # Load logo
                original_logo = self.generator.load_real_logo(self.logo_mappings[class_name])
                if original_logo is None:
                    logger.error(f"Failed to load logo for {class_name}")
                    continue
                
                success_count = 0
                
                for i in range(images_per_class):
                    try:
                        import random
                        # 90%はオリジナルの色、10%は色変更
                        use_original = not self.generator.should_change_color()
                        
                        variation_params = {
                            'use_original_color': use_original,
                            'background': random.choice(self.generator.background_types),
                            'rotation': self.generator.get_rotation_angle(),
                            'size_ratio': random.choice(self.generator.logo_size_ratios)
                        }
                        
                        # 色変更する場合のみ、色を指定
                        if not use_original:
                            variation_params['color'] = random.choice(self.generator.logo_colors)
                        
                        canvas_sizes = [(640, 480), (800, 600), (1024, 768), (1280, 720)]
                        canvas_size = random.choice(canvas_sizes)
                        
                        variation_image, bbox = self.generator.generate_logo_variation(
                            original_logo, canvas_size, variation_params
                        )
                        
                        temp_dir = Path("temp_logo_variations")
                        temp_dir.mkdir(exist_ok=True)
                        temp_path = temp_dir / f"{class_name}_variation_{i:04d}.jpg"
                        
                        variation_image.save(temp_path, "JPEG", quality=90)
                        
                        import requests
                        add_payload = {
                            "image_path": str(temp_path),
                            "annotations": [{
                                "class_name": class_name,
                                "bbox": bbox,
                                "confidence": 1.0
                            }],
                            "split": "train" if i < images_per_class * 0.8 else "val",
                            "preserve_filename": True  # 元のファイル名を保持
                        }
                        
                        response = requests.post(
                            f"{self.generator.base_url}/training/datasets/{dataset_name}/add-image",
                            headers={"Content-Type": "application/json"},
                            json=add_payload
                        )
                        
                        if response.status_code == 200:
                            success_count += 1
                        
                        if temp_path.exists():
                            temp_path.unlink()
                            
                    except Exception as e:
                        logger.error(f"Error generating variation {i} for {class_name}: {e}")
                        continue
                
                logger.info(f"Generated {success_count}/{images_per_class} variations for {class_name}")
            
            # Cleanup
            temp_dir = Path("temp_logo_variations")
            if temp_dir.exists():
                try:
                    temp_dir.rmdir()
                except:
                    pass
            
            return {
                "success": True,
                "message": f"Dataset '{dataset_name}' created with real logo images",
                "dataset_name": dataset_name,
                "classes": class_names,
                "images_per_class": images_per_class,
                "total_images": len(class_names) * images_per_class
            }
            
        except Exception as e:
            logger.error(f"Failed to generate real logo dataset: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Singleton instance
_real_logo_adapter = None

def get_real_logo_adapter() -> RealLogoAdapter:
    """Get singleton instance of RealLogoAdapter."""
    global _real_logo_adapter
    if _real_logo_adapter is None:
        _real_logo_adapter = RealLogoAdapter()
    return _real_logo_adapter