import json
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BrandClassificationError(Exception):
    """Exception raised by brand classifier."""
    pass


class BrandClassifier:
    """Brand classification and normalization utility."""
    
    def __init__(self):
        self.settings = get_settings()
        self.brand_mapping: Dict = {}
        self.logo_categories: Dict = {}
        self._load_classification_data()
    
    def _load_classification_data(self):
        """Load brand mapping and category data."""
        try:
            # Load brand mapping
            brand_mapping_path = Path(self.settings.brand_mapping_file)
            if brand_mapping_path.exists():
                with open(brand_mapping_path, 'r', encoding='utf-8') as f:
                    self.brand_mapping = json.load(f)
                logger.info(f"Loaded brand mapping from {brand_mapping_path}")
            else:
                logger.warning(f"Brand mapping file not found: {brand_mapping_path}")
                self.brand_mapping = {"brand_normalization": {}}
            
            # Load logo categories
            categories_path = Path(self.settings.logo_categories_file)
            if categories_path.exists():
                with open(categories_path, 'r', encoding='utf-8') as f:
                    self.logo_categories = json.load(f)
                logger.info(f"Loaded logo categories from {categories_path}")
            else:
                logger.warning(f"Logo categories file not found: {categories_path}")
                self.logo_categories = {"categories": {}, "confidence_adjustments": {}}
                
        except Exception as e:
            logger.error(f"Failed to load classification data: {e}")
            self.brand_mapping = {"brand_normalization": {}}
            self.logo_categories = {"categories": {}, "confidence_adjustments": {}}
    
    def normalize_brand_name(self, detected_name: str) -> Dict[str, str]:
        """
        Normalize detected brand name to standard format.
        
        Args:
            detected_name: Raw detected brand name
            
        Returns:
            Dictionary with normalized brand information
        """
        try:
            # Clean the detected name
            clean_name = detected_name.upper().strip()
            
            # Direct lookup
            brand_data = self.brand_mapping.get("brand_normalization", {}).get(clean_name)
            if brand_data:
                return {
                    "original": detected_name,
                    "normalized": clean_name,
                    "japanese": brand_data.get("japanese", clean_name),
                    "english": brand_data.get("english", clean_name),
                    "official_name": brand_data.get("official_name", clean_name),
                    "aliases": brand_data.get("aliases", [clean_name])
                }
            
            # Fuzzy matching through aliases
            for brand_key, brand_info in self.brand_mapping.get("brand_normalization", {}).items():
                aliases = brand_info.get("aliases", [])
                if any(alias.upper() == clean_name for alias in aliases):
                    return {
                        "original": detected_name,
                        "normalized": brand_key,
                        "japanese": brand_info.get("japanese", brand_key),
                        "english": brand_info.get("english", brand_key),
                        "official_name": brand_info.get("official_name", brand_key),
                        "aliases": brand_info.get("aliases", [brand_key])
                    }
            
            # No match found - return original with minimal processing
            return {
                "original": detected_name,
                "normalized": clean_name,
                "japanese": clean_name,
                "english": clean_name,
                "official_name": clean_name,
                "aliases": [clean_name]
            }
            
        except Exception as e:
            logger.warning(f"Failed to normalize brand name '{detected_name}': {e}")
            return {
                "original": detected_name,
                "normalized": detected_name,
                "japanese": detected_name,
                "english": detected_name,
                "official_name": detected_name,
                "aliases": [detected_name]
            }
    
    def get_brand_category(self, brand_name: str) -> Optional[Dict[str, str]]:
        """
        Get category information for a brand.
        
        Args:
            brand_name: Normalized brand name
            
        Returns:
            Category information or None if not found
        """
        try:
            clean_name = brand_name.upper().strip()
            
            for category_key, category_data in self.logo_categories.get("categories", {}).items():
                brands = category_data.get("brands", [])
                if clean_name in brands:
                    # Check for subcategory
                    subcategory_info = None
                    for subcat_key, subcat_data in category_data.get("subcategories", {}).items():
                        if clean_name in subcat_data.get("brands", []):
                            subcategory_info = {
                                "key": subcat_key,
                                "name": subcat_data.get("name", subcat_key),
                                "name_en": subcat_data.get("name_en", subcat_key)
                            }
                            break
                    
                    return {
                        "category": {
                            "key": category_key,
                            "name": category_data.get("name", category_key),
                            "name_en": category_data.get("name_en", category_key)
                        },
                        "subcategory": subcategory_info
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get category for brand '{brand_name}': {e}")
            return None
    
    def adjust_confidence_by_category(self, brand_name: str, original_confidence: float) -> float:
        """
        Adjust confidence score based on brand category.
        
        Args:
            brand_name: Normalized brand name
            original_confidence: Original confidence score
            
        Returns:
            Adjusted confidence score
        """
        try:
            category_info = self.get_brand_category(brand_name)
            if not category_info:
                return original_confidence
            
            category_key = category_info["category"]["key"]
            adjustments = self.logo_categories.get("confidence_adjustments", {})
            
            if category_key in adjustments:
                adjustment = adjustments[category_key].get("threshold_adjustment", 0)
                adjusted_confidence = max(0.0, min(1.0, original_confidence + adjustment))
                
                logger.debug(
                    f"Adjusted confidence for {brand_name} in category {category_key}: "
                    f"{original_confidence:.3f} -> {adjusted_confidence:.3f} ({adjustment:+.3f})"
                )
                
                return adjusted_confidence
            
            return original_confidence
            
        except Exception as e:
            logger.warning(f"Failed to adjust confidence for brand '{brand_name}': {e}")
            return original_confidence
    
    def get_available_brands(self) -> List[Dict[str, str]]:
        """Get list of all available brands."""
        try:
            brands = []
            for brand_key, brand_data in self.brand_mapping.get("brand_normalization", {}).items():
                category_info = self.get_brand_category(brand_key)
                brands.append({
                    "key": brand_key,
                    "japanese": brand_data.get("japanese", brand_key),
                    "english": brand_data.get("english", brand_key),
                    "official_name": brand_data.get("official_name", brand_key),
                    "category": category_info["category"]["name"] if category_info else "その他",
                    "category_en": category_info["category"]["name_en"] if category_info else "Other"
                })
            
            return sorted(brands, key=lambda x: x["japanese"])
            
        except Exception as e:
            logger.error(f"Failed to get available brands: {e}")
            return []
    
    def get_available_categories(self) -> List[Dict[str, any]]:
        """Get list of all available categories."""
        try:
            categories = []
            for category_key, category_data in self.logo_categories.get("categories", {}).items():
                subcategories = []
                for subcat_key, subcat_data in category_data.get("subcategories", {}).items():
                    subcategories.append({
                        "key": subcat_key,
                        "name": subcat_data.get("name", subcat_key),
                        "name_en": subcat_data.get("name_en", subcat_key),
                        "brand_count": len(subcat_data.get("brands", []))
                    })
                
                categories.append({
                    "key": category_key,
                    "name": category_data.get("name", category_key),
                    "name_en": category_data.get("name_en", category_key),
                    "brand_count": len(category_data.get("brands", [])),
                    "subcategories": subcategories
                })
            
            return sorted(categories, key=lambda x: x["name"])
            
        except Exception as e:
            logger.error(f"Failed to get available categories: {e}")
            return []
    
    def reload_data(self):
        """Reload classification data from files."""
        logger.info("Reloading brand classification data...")
        self._load_classification_data()


# Global brand classifier instance
_brand_classifier: Optional[BrandClassifier] = None


def get_brand_classifier() -> BrandClassifier:
    """Get the global brand classifier instance (singleton)."""
    global _brand_classifier
    
    if _brand_classifier is None:
        _brand_classifier = BrandClassifier()
    
    return _brand_classifier