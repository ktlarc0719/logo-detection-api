import asyncio
import aiohttp
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from io import BytesIO
import tempfile
import os

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ImageDownloadError(Exception):
    """Exception raised when image download fails."""
    pass


class ImageProcessingError(Exception):
    """Exception raised when image processing fails."""
    pass


async def download_image(
    session: aiohttp.ClientSession,
    url: str,
    timeout: Optional[int] = None
) -> bytes:
    """Download image from URL asynchronously."""
    settings = get_settings()
    timeout = timeout or settings.download_timeout
    
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            if response.status != 200:
                raise ImageDownloadError(
                    f"HTTP {response.status}: Failed to download image from {url}"
                )
            
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > settings.max_request_size:
                raise ImageDownloadError(
                    f"Image too large: {content_length} bytes (max: {settings.max_request_size})"
                )
            
            return await response.read()
    
    except asyncio.TimeoutError:
        raise ImageDownloadError(f"Timeout downloading image from {url}")
    except aiohttp.ClientError as e:
        raise ImageDownloadError(f"Client error downloading image from {url}: {str(e)}")
    except Exception as e:
        raise ImageDownloadError(f"Unexpected error downloading image from {url}: {str(e)}")


def validate_image_format(image_data: bytes) -> bool:
    """Validate if image data is in supported format."""
    settings = get_settings()
    
    try:
        with Image.open(BytesIO(image_data)) as img:
            format_lower = img.format.lower() if img.format else ""
            return format_lower in settings.supported_formats
    except Exception:
        return False


def preprocess_image(image_data: bytes) -> np.ndarray:
    """Preprocess image data for YOLO detection."""
    settings = get_settings()
    
    try:
        # Convert bytes to PIL Image
        pil_image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Resize if too large
        width, height = pil_image.size
        if max(width, height) > settings.max_image_size:
            ratio = settings.max_image_size / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to numpy array (RGB format for YOLO)
        image_array = np.array(pil_image)
        
        return image_array
    
    except Exception as e:
        raise ImageProcessingError(f"Failed to preprocess image: {str(e)}")


def save_temp_image(image_data: bytes, suffix: str = ".jpg") -> str:
    """Save image data to temporary file and return path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(image_data)
            return temp_file.name
    except Exception as e:
        raise ImageProcessingError(f"Failed to save temporary image: {str(e)}")


def cleanup_temp_file(file_path: str) -> None:
    """Clean up temporary file."""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {str(e)}")


def get_image_info(image_data: bytes) -> dict:
    """Get basic image information."""
    try:
        with Image.open(BytesIO(image_data)) as img:
            return {
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "width": img.width,
                "height": img.height,
                "has_transparency": img.mode in ("RGBA", "LA") or "transparency" in img.info
            }
    except Exception as e:
        raise ImageProcessingError(f"Failed to get image info: {str(e)}")


class ImageProcessor:
    """Utility class for batch image processing."""
    
    def __init__(self):
        self.settings = get_settings()
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(
            limit=self.settings.max_concurrent_downloads,
            limit_per_host=20
        )
        self.session = aiohttp.ClientSession(connector=connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def process_image_url(self, url: str) -> Tuple[np.ndarray, dict]:
        """Download and process single image URL."""
        if not self.session:
            raise RuntimeError("ImageProcessor must be used as async context manager")
        
        # Download image
        image_data = await download_image(self.session, url)
        
        # Validate format
        if not validate_image_format(image_data):
            raise ImageProcessingError(f"Unsupported image format from {url}")
        
        # Get image info
        image_info = get_image_info(image_data)
        
        # Preprocess for detection
        processed_image = preprocess_image(image_data)
        
        return processed_image, image_info