from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from typing import Dict, Any, Optional
import asyncio
import aiohttp
import hashlib
from pathlib import Path
import logging
import os

from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/download",
    tags=["download"]
)


async def download_image(session: aiohttp.ClientSession, url: str, save_path: Path) -> Dict[str, Any]:
    """Download a single image from URL."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                content = await response.read()
                
                # Generate filename from URL hash
                url_hash = hashlib.md5(url.encode()).hexdigest()
                ext = url.split('.')[-1].split('?')[0]  # Extract extension
                if ext not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                    ext = 'jpg'  # Default extension
                
                filename = f"{url_hash}.{ext}"
                file_path = save_path / filename
                
                # Save image
                with open(file_path, 'wb') as f:
                    f.write(content)
                
                return {
                    "url": url,
                    "status": "success",
                    "filename": filename,
                    "size": len(content)
                }
            else:
                return {
                    "url": url,
                    "status": "failed",
                    "error": f"HTTP {response.status}"
                }
    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        return {
            "url": url,
            "status": "failed",
            "error": str(e)
        }


async def download_images_task(urls_file: Path, download_dir: Path, create_dir: bool = False) -> Dict[str, Any]:
    """Background task to download images from URLs file."""
    try:
        # Read URLs from file
        if not urls_file.exists():
            return {
                "status": "error",
                "message": f"URLs file not found: {urls_file}"
            }
        
        with open(urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(urls))
        
        # Check if download directory exists
        if not download_dir.exists():
            if create_dir:
                try:
                    download_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {download_dir}")
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Failed to create directory: {str(e)}"
                    }
            else:
                return {
                    "status": "error",
                    "message": f"Directory does not exist: {download_dir}"
                }
        
        # Check existing files
        existing_files = set(f.name for f in download_dir.glob('*'))
        
        # Filter out already downloaded URLs
        urls_to_download = []
        skipped = []
        
        for url in unique_urls:
            url_hash = hashlib.md5(url.encode()).hexdigest()
            # Check if any file with this hash exists (regardless of extension)
            if any(f.startswith(url_hash) for f in existing_files):
                skipped.append(url)
            else:
                urls_to_download.append(url)
        
        logger.info(f"Total unique URLs: {len(unique_urls)}, To download: {len(urls_to_download)}, Skipped: {len(skipped)}")
        
        # Download images concurrently
        results = []
        async with aiohttp.ClientSession() as session:
            tasks = [download_image(session, url, download_dir) for url in urls_to_download]
            results = await asyncio.gather(*tasks)
        
        # Count successful downloads
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "failed")
        
        return {
            "status": "completed",
            "total_urls": len(urls),
            "unique_urls": len(unique_urls),
            "downloaded": successful,
            "failed": failed,
            "skipped": len(skipped),
            "duplicates_removed": len(urls) - len(unique_urls),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Download task failed: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/images")
async def download_images_from_file(
    background_tasks: BackgroundTasks,
    subdirectory: Optional[str] = Query(None, description="Subdirectory name (e.g., 'bandai')"),
    create_dir: bool = Query(False, description="Create directory if it doesn't exist")
) -> Dict[str, Any]:
    """
    Download images from URLs listed in annotation_work/download_urls.txt.
    
    - Removes duplicate URLs
    - Skips already downloaded images
    - Downloads to specified Windows directory or default location
    
    Parameters:
    - subdirectory: Optional subdirectory name (e.g., 'bandai')
    - create_dir: Whether to create the directory if it doesn't exist
    """
    urls_file = Path("annotation_work/download_urls.txt")
    
    # Determine download directory
    if subdirectory:
        # WSL path to Windows directory
        windows_base_path = Path("/mnt/c/01_annotation")
        download_dir = windows_base_path / subdirectory / "images"
    else:
        # Default path
        download_dir = Path("annotation_work/downloaded_images")
    
    # Check if URLs file exists
    if not urls_file.exists():
        # Try alternative name
        urls_file = Path("annotation_work/download.txt")
        if not urls_file.exists():
            return {
                "status": "error",
                "message": "URLs file not found. Expected: annotation_work/download_urls.txt or annotation_work/download.txt"
            }
    
    # Run download in background
    result = await download_images_task(urls_file, download_dir, create_dir)
    
    return result


@router.get("/images/status")
async def get_download_status(
    subdirectory: Optional[str] = Query(None, description="Subdirectory name (e.g., 'bandai')")
) -> Dict[str, Any]:
    """Check the status of downloaded images."""
    # Determine download directory
    if subdirectory:
        windows_base_path = Path("/mnt/c/01_annotation")
        download_dir = windows_base_path / subdirectory / "images"
    else:
        download_dir = Path("annotation_work/downloaded_images")
    
    urls_file = Path("annotation_work/download_urls.txt")
    
    if not urls_file.exists():
        urls_file = Path("annotation_work/download.txt")
    
    # Count downloaded files
    downloaded_files = list(download_dir.glob('*')) if download_dir.exists() else []
    
    # Count total URLs
    total_urls = 0
    unique_urls = 0
    if urls_file.exists():
        with open(urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
            total_urls = len(urls)
            unique_urls = len(set(urls))
    
    return {
        "urls_file": str(urls_file),
        "download_directory": str(download_dir),
        "total_urls_in_file": total_urls,
        "unique_urls": unique_urls,
        "downloaded_files": len(downloaded_files),
        "files": [f.name for f in downloaded_files]
    }