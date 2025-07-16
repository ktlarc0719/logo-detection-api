"""
画像ファイルキャッシュシステム
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """キャッシュエントリー"""
    files: Set[str]  # ASINのセット
    file_paths: Dict[str, str]  # ASIN -> フルパスのマッピング
    timestamp: float
    scan_time: float  # スキャンにかかった時間


class ImageFileCache:
    """画像ファイルキャッシュ"""
    
    def __init__(self, cache_ttl: int = 300):  # デフォルト5分
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_ttl = cache_ttl
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._base_path = "/mnt/c/03_amazon_images"
        
    async def get_image_files(self, seller_id: str) -> Tuple[Set[str], Dict[str, str]]:
        """
        セラーIDの画像ファイルを取得
        Returns:
            Tuple[Set[str], Dict[str, str]]: (ASINのセット, ASIN->パスのマッピング)
        """
        async with self._lock:
            cache_key = seller_id
            current_time = time.time()
            
            # キャッシュチェック
            if cache_key in self._cache:
                cached_data = self._cache[cache_key]
                if current_time - cached_data.timestamp < self._cache_ttl:
                    logger.info(f"Cache hit for seller {seller_id} (age: {current_time - cached_data.timestamp:.1f}s)")
                    return cached_data.files, cached_data.file_paths
                else:
                    logger.info(f"Cache expired for seller {seller_id}")
            
        # キャッシュなし/期限切れの場合はスキャン
        start_time = time.time()
        files, file_paths = await self._scan_seller_directory(seller_id)
        scan_time = time.time() - start_time
        
        # キャッシュに保存
        async with self._lock:
            self._cache[cache_key] = CacheEntry(
                files=files,
                file_paths=file_paths,
                timestamp=time.time(),
                scan_time=scan_time
            )
        
        logger.info(f"Scanned {len(files)} files for seller {seller_id} in {scan_time:.2f}s")
        return files, file_paths
        
    async def _scan_seller_directory(self, seller_id: str) -> Tuple[Set[str], Dict[str, str]]:
        """セラーディレクトリをスキャン"""
        seller_path = Path(self._base_path) / seller_id
        
        if not seller_path.exists():
            logger.warning(f"Seller directory not found: {seller_path}")
            return set(), {}
            
        # ThreadPoolExecutorで同期的なファイルシステム操作を実行
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._scan_directory_sync,
            seller_path
        )
        
    def _scan_directory_sync(self, seller_path: Path) -> Tuple[Set[str], Dict[str, str]]:
        """同期的にディレクトリをスキャン"""
        asins = set()
        file_paths = {}
        
        # サブディレクトリ（0-9, A-Z）をスキャン
        for subdir in seller_path.iterdir():
            if not subdir.is_dir():
                continue
                
            # サブディレクトリ名が1文字でない場合はスキップ
            if len(subdir.name) != 1:
                continue
                
            # 画像ファイルを検索
            for file_path in subdir.glob("*.png"):
                asin = file_path.stem  # 拡張子を除いたファイル名
                asins.add(asin)
                file_paths[asin] = str(file_path)
                
        return asins, file_paths
        
    async def invalidate_cache(self, seller_id: Optional[str] = None):
        """キャッシュを無効化"""
        async with self._lock:
            if seller_id:
                if seller_id in self._cache:
                    del self._cache[seller_id]
                    logger.info(f"Invalidated cache for seller {seller_id}")
            else:
                self._cache.clear()
                logger.info("Invalidated all cache entries")
                
    async def get_cache_stats(self) -> Dict[str, any]:
        """キャッシュ統計を取得"""
        async with self._lock:
            total_entries = len(self._cache)
            total_files = sum(len(entry.files) for entry in self._cache.values())
            avg_scan_time = (
                sum(entry.scan_time for entry in self._cache.values()) / total_entries
                if total_entries > 0 else 0
            )
            
            return {
                "total_entries": total_entries,
                "total_files_cached": total_files,
                "average_scan_time": avg_scan_time,
                "cache_ttl": self._cache_ttl,
                "cached_sellers": list(self._cache.keys())
            }


# シングルトンインスタンス
_image_file_cache: Optional[ImageFileCache] = None


def get_image_file_cache() -> ImageFileCache:
    """画像ファイルキャッシュのシングルトンインスタンスを取得"""
    global _image_file_cache
    if _image_file_cache is None:
        _image_file_cache = ImageFileCache()
    return _image_file_cache