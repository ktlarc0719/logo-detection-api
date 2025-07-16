"""
画像検査エンジン
"""

import os
import asyncio
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import torch
import glob
import random

from src.core.detection_engine import get_detection_engine
from src.models.inspection_schemas import (
    InspectionRequest, InspectionItem, InspectionResult,
    InspectionBatchResult, InspectionStatus, InspectionMode
)
from src.utils.logger import get_logger
from src.db.dummy_db import get_dummy_inspection_items, save_inspection_result

logger = get_logger(__name__)


class InspectionEngine:
    """画像検査エンジン"""
    
    def __init__(self):
        self.detection_engine = get_detection_engine()
        self.active_batches: Dict[str, InspectionStatus] = {}
        self.batch_results: Dict[str, InspectionBatchResult] = {}  # Store batch results
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._gpu_load_manager = GPULoadManager()
        self.device_mode = "cpu"  # Default to CPU mode
        self.windows_base_path = "/mnt/c/03_amazon_images"  # Windows path for seller images
        
    def generate_image_path(self, base_path: str, seller_id: str, asin: str) -> str:
        """
        画像パスを生成
        Format: basePath/{sellerId}/{asinの末尾1文字}/{ASIN}.png
        """
        last_char = asin[-1] if asin else '0'
        return os.path.join(base_path, seller_id, last_char, f"{asin}.png")
    
    async def process_inspection_batch(self, request: InspectionRequest) -> str:
        """
        検査バッチを処理（非同期）
        
        Returns:
            batch_id: バッチ処理ID
        """
        batch_id = str(uuid.uuid4())
        
        # ステータスを初期化
        self.active_batches[batch_id] = InspectionStatus(
            batch_id=batch_id,
            status="initializing",
            progress=0.0,
            items_processed=0,
            items_total=0,
            created_at=datetime.utcnow(),
            model_name=request.model_name,
            start_time=datetime.utcnow()
        )
        
        # バックグラウンドで処理開始
        asyncio.create_task(self._process_batch_async(batch_id, request))
        
        return batch_id
    
    async def _process_batch_async(self, batch_id: str, request: InspectionRequest):
        """バッチ処理の実行（非同期）"""
        try:
            # ステータス更新
            self.active_batches[batch_id].status = "loading_items"
            
            # デバイスモードを設定
            if hasattr(request, 'device_mode'):
                self.set_device_mode(request.device_mode)
            
            # 検査対象アイテムを取得
            items = await self._get_inspection_items(request)
            
            if not items:
                raise ValueError("No items to process")
            
            # ステータス更新
            self.active_batches[batch_id].items_total = len(items)
            self.active_batches[batch_id].status = "running"
            
            # モデルを切り替え
            if request.model_name:
                self.detection_engine.switch_model(request.model_name)
            
            # バッチ結果を初期化
            batch_result = InspectionBatchResult(
                batch_id=batch_id,
                mode=request.mode,
                total_items=len(items),
                processed_items=0,
                successful_items=0,
                failed_items=0,
                results=[],
                model_used=request.model_name,
                gpu_load_rate=request.gpu_load_rate,
                start_time=datetime.utcnow()
            )
            
            # GPU負荷管理を設定
            self._gpu_load_manager.set_load_rate(request.gpu_load_rate)
            
            # 各アイテムを処理
            for i, item in enumerate(items):
                try:
                    # GPU負荷調整
                    await self._gpu_load_manager.wait_if_needed()
                    
                    # ステータス更新
                    self.active_batches[batch_id].current_item = item.asin
                    self.active_batches[batch_id].items_processed = i
                    self.active_batches[batch_id].progress = (i / len(items)) * 100
                    
                    # 画像検査実行
                    result = await self._inspect_single_item(
                        item,
                        request.confidence_threshold,
                        request.max_detections
                    )
                    
                    batch_result.results.append(result)
                    batch_result.processed_items += 1
                    
                    if result.error:
                        batch_result.failed_items += 1
                    else:
                        batch_result.successful_items += 1
                        
                        # DBに結果を保存
                        await save_inspection_result(result)
                    
                except Exception as e:
                    logger.error(f"Error processing item {item.asin}: {e}")
                    batch_result.failed_items += 1
                    self.active_batches[batch_id].errors.append(str(e))
            
            # 処理完了
            batch_result.end_time = datetime.utcnow()
            batch_result.total_processing_time = (
                batch_result.end_time - batch_result.start_time
            ).total_seconds()
            
            self.active_batches[batch_id].status = "completed"
            self.active_batches[batch_id].progress = 100.0
            self.active_batches[batch_id].end_time = batch_result.end_time
            self.active_batches[batch_id].elapsed_time = batch_result.total_processing_time
            
            # Calculate images per hour
            if batch_result.total_processing_time > 0:
                self.active_batches[batch_id].images_per_hour = (
                    batch_result.processed_items / batch_result.total_processing_time * 3600
                )
            
            # 結果を保存
            await self._save_batch_result(batch_result)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            self.active_batches[batch_id].status = "failed"
            self.active_batches[batch_id].errors.append(str(e))
    
    async def _get_inspection_items(self, request: InspectionRequest) -> List[InspectionItem]:
        """検査対象アイテムを取得"""
        
        if request.mode == InspectionMode.INDIVIDUAL and request.seller_id:
            # セラーIDモード: Windowsパスから画像を取得
            items = await self._get_seller_images(request.seller_id, request.max_items)
        else:
            # 従来の個別指定モードまたは全件実行モード
            base_path = "/data/product_images"  # 設定から取得
            
            if request.mode == InspectionMode.INDIVIDUAL:
                items = await get_dummy_inspection_items(
                    seller_id=request.seller_id,
                    user_id=request.user_id,
                    limit=request.max_items
                )
            else:
                items = await get_dummy_inspection_items(
                    limit=request.max_items if not request.process_all else None
                )
            
            # 画像パスを生成
            for item in items:
                item.image_path = self.generate_image_path(
                    base_path,
                    item.seller_id,
                    item.asin
                )
        
        return items
    
    async def _inspect_single_item(
        self,
        item: InspectionItem,
        confidence_threshold: float,
        max_detections: int
    ) -> InspectionResult:
        """単一アイテムの検査"""
        start_time = time.time()
        
        try:
            # 画像を読み込み
            if not os.path.exists(item.image_path):
                # ダミー実装: ランダム画像を生成
                image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            else:
                image = cv2.imread(item.image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 検出実行
            detections = self.detection_engine.detect(
                image,
                confidence_threshold=confidence_threshold,
                max_detections=max_detections
            )
            
            # 結果を整理
            result = InspectionResult(
                item_id=str(uuid.uuid4()),
                seller_id=item.seller_id,
                asin=item.asin,
                user_id=item.user_id,
                image_path=item.image_path,
                detected=len(detections) > 0,
                detections=[det.dict() for det in detections],
                confidence_scores=[det.confidence for det in detections],
                labels=[det.logo_name for det in detections],
                model_used=self.detection_engine.current_model_name,
                processing_time=time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Inspection failed for {item.asin}: {e}")
            return InspectionResult(
                item_id=str(uuid.uuid4()),
                seller_id=item.seller_id,
                asin=item.asin,
                user_id=item.user_id,
                image_path=item.image_path,
                detected=False,
                model_used=self.detection_engine.current_model_name,
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _save_batch_result(self, batch_result: InspectionBatchResult):
        """バッチ結果を保存"""
        # メモリに結果を保存（実際のアプリケーションではDBに保存）
        self.batch_results[batch_result.batch_id] = batch_result
        logger.info(f"Batch {batch_result.batch_id} completed: "
                   f"{batch_result.successful_items}/{batch_result.total_items} successful")
    
    def get_batch_results(self, batch_id: str) -> Optional[InspectionBatchResult]:
        """バッチ結果を取得"""
        return self.batch_results.get(batch_id)
    
    def get_batch_status(self, batch_id: str) -> Optional[InspectionStatus]:
        """バッチのステータスを取得"""
        return self.active_batches.get(batch_id)
    
    def get_all_batch_statuses(self) -> List[InspectionStatus]:
        """全てのバッチステータスを取得（新しい順）"""
        # Sort by created_at in descending order (newest first)
        return sorted(
            self.active_batches.values(),
            key=lambda x: x.created_at,
            reverse=True
        )
    
    def cancel_batch(self, batch_id: str) -> bool:
        """バッチをキャンセル"""
        if batch_id in self.active_batches:
            self.active_batches[batch_id].status = "cancelled"
            return True
        return False
    
    def set_device_mode(self, mode: str):
        """デバイスモードを設定 (cpu/gpu)"""
        if mode in ["cpu", "gpu"]:
            self.device_mode = mode
            # detection_engineのデバイスも更新
            if mode == "gpu" and torch.cuda.is_available():
                self.detection_engine.device = "cuda"
            else:
                self.detection_engine.device = "cpu"
            logger.info(f"Device mode set to: {self.detection_engine.device}")
    
    async def _get_seller_images(self, seller_id: str, max_items: Optional[int]) -> List[InspectionItem]:
        """セラーIDに基づいてWindowsパスから画像を取得"""
        items = []
        seller_path = os.path.join(self.windows_base_path, seller_id)
        
        if not os.path.exists(seller_path):
            logger.warning(f"Seller path not found: {seller_path}")
            return items
        
        # 各サブフォルダから画像を収集
        image_files = []
        for subfolder in os.listdir(seller_path):
            subfolder_path = os.path.join(seller_path, subfolder)
            if os.path.isdir(subfolder_path):
                # 画像ファイルを検索
                patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
                for pattern in patterns:
                    image_files.extend(glob.glob(os.path.join(subfolder_path, pattern)))
        
        # ランダムにシャッフル
        random.shuffle(image_files)
        
        # max_itemsに従って制限
        if max_items:
            image_files = image_files[:max_items]
        
        # InspectionItemオブジェクトを作成
        for image_path in image_files:
            # ファイル名からASINを抽出（拡張子を除く）
            filename = os.path.basename(image_path)
            asin = os.path.splitext(filename)[0]
            
            item = InspectionItem(
                seller_id=seller_id,
                asin=asin,
                image_path=image_path,
                status="pending"
            )
            items.append(item)
        
        logger.info(f"Found {len(items)} images for seller {seller_id}")
        return items


class GPULoadManager:
    """GPU負荷管理"""
    
    def __init__(self):
        self.load_rate = 0.8
        self.last_process_time = time.time()
        self.min_interval = 0.01  # 最小間隔（秒）
    
    def set_load_rate(self, rate: float):
        """負荷率を設定（0.1-1.0）"""
        self.load_rate = max(0.1, min(1.0, rate))
        # 負荷率に応じて最小間隔を調整
        self.min_interval = (1.0 - self.load_rate) * 0.1
    
    async def wait_if_needed(self):
        """必要に応じて待機"""
        elapsed = time.time() - self.last_process_time
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_process_time = time.time()
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """GPU状態を取得"""
        if torch.cuda.is_available():
            return {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated": torch.cuda.memory_allocated(0) / 1e9,
                "memory_reserved": torch.cuda.memory_reserved(0) / 1e9,
                "load_rate": self.load_rate
            }
        else:
            return {
                "available": False,
                "load_rate": self.load_rate
            }


# シングルトンインスタンス
_inspection_engine: Optional[InspectionEngine] = None


def get_inspection_engine() -> InspectionEngine:
    """検査エンジンのインスタンスを取得"""
    global _inspection_engine
    if _inspection_engine is None:
        _inspection_engine = InspectionEngine()
    return _inspection_engine