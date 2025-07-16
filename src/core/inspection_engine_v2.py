"""
画像検査エンジン V2 - DB対応版
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

from src.core.detection_engine import get_detection_engine
from src.core.image_file_cache import get_image_file_cache
from src.models.inspection_schemas_v2 import (
    InspectionRequest, InspectionItem, InspectionResult,
    InspectionBatchResult, InspectionStatus, InspectionMode
)
from src.db.inspection_repository import inspection_repository
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GPULoadManager:
    """GPU負荷管理クラス"""
    def __init__(self):
        self.gpu_load_rate = 0.8
        self.is_gpu_available = torch.cuda.is_available()
        
    def set_load_rate(self, rate: float):
        """GPU負荷率を設定 (0.0-1.0)"""
        self.gpu_load_rate = max(0.0, min(1.0, rate))
        
    def get_batch_size(self, device_mode: str) -> int:
        """デバイスモードに応じたバッチサイズを取得"""
        if device_mode == "gpu" and self.is_gpu_available:
            # GPU使用時: 負荷率に応じて調整
            base_size = 48
            return int(base_size * self.gpu_load_rate)
        else:
            # CPU使用時
            return 12
            
    def get_parallel_processes(self, device_mode: str) -> int:
        """デバイスモードに応じた並列プロセス数を取得"""
        if device_mode == "gpu" and self.is_gpu_available:
            # GPU使用時: 負荷率に応じて調整
            base_processes = 3
            return max(1, int(base_processes * self.gpu_load_rate))
        else:
            # CPU使用時
            return 10


class InspectionEngineV2:
    """画像検査エンジン V2"""
    
    def __init__(self):
        self.detection_engine = get_detection_engine()
        self.active_batches: Dict[str, InspectionStatus] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._gpu_load_manager = GPULoadManager()
        self.device_mode = "cpu"  # Default to CPU mode
        
    async def start_inspection(self, request: InspectionRequest) -> str:
        """検査を開始"""
        try:
            # アイテムを取得
            items = await self._get_inspection_items(request)
            
            if not items:
                raise ValueError("No items found for inspection")
            
            # バッチをDBに作成
            batch_id = await inspection_repository.create_batch(request, len(items))
            
            # ステータスを作成
            status = InspectionStatus(
                batch_id=batch_id,
                status="running",
                total_items=len(items),
                processed_items=0,
                detected_items=0,
                failed_items=0,
                progress=0.0,
                start_time=datetime.utcnow(),
                mode=request.mode,
                seller_ids=request.sellers,
                base_path=request.base_path
            )
            
            self.active_batches[batch_id] = status
            
            # 非同期でバッチ処理を開始
            asyncio.create_task(self._process_batch(batch_id, items, request))
            
            return batch_id
            
        except Exception as e:
            logger.error(f"Failed to start inspection: {e}")
            raise
    
    async def _get_inspection_items(self, request: InspectionRequest) -> List[InspectionItem]:
        """検査対象アイテムを取得"""
        items = []
        
        if request.mode == InspectionMode.SELLER:
            # ファイルキャッシュを取得
            file_cache = get_image_file_cache()
            
            # DBからセラーIDに基づいてアイテムを取得
            for seller_id in request.sellers:
                # まずセラーの画像ファイルをキャッシュから取得
                asins_with_images, file_paths = await file_cache.get_image_files(seller_id)
                logger.info(f"Found {len(asins_with_images)} image files for seller {seller_id}")
                
                # DBからアイテムを取得
                db_items = await inspection_repository.get_items_from_inventory(
                    seller_id,
                    request.max_items,
                    only_uninspected=request.only_uninspected,
                    model_used=request.model_name
                )
                
                # 画像が存在するアイテムのみをフィルタリング
                for item in db_items:
                    asin = item['asin']
                    if asin in asins_with_images:
                        # 実際のファイルパスを使用
                        items.append(InspectionItem(
                            item_id=str(item['id']),
                            seller_id=item['seller_id'],
                            asin=asin,
                            image_path=file_paths[asin]
                        ))
                    else:
                        logger.debug(f"Image not found for ASIN: {asin}")
                    
        elif request.mode == InspectionMode.PATH:
            # 絶対パスからアイテムを取得
            if not request.base_path:
                raise ValueError("base_path is required for path mode")
                
            base_path = Path(request.base_path)
            if not base_path.exists():
                raise ValueError(f"Path does not exist: {request.base_path}")
            
            # 画像ファイルを検索
            patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
            image_files = []
            
            if request.include_subdirs:
                # サブディレクトリを含む
                for pattern in patterns:
                    image_files.extend(base_path.rglob(pattern))
            else:
                # 直下のみ
                for pattern in patterns:
                    image_files.extend(base_path.glob(pattern))
            
            # 最大数を制限
            if request.max_items:
                image_files = image_files[:request.max_items]
            
            # InspectionItemに変換
            for img_path in image_files:
                items.append(InspectionItem(
                    item_id=str(uuid.uuid4()),
                    image_path=str(img_path),
                    seller_id=None,
                    asin=None
                ))
        
        return items
    
    async def _process_batch(
        self,
        batch_id: str,
        items: List[InspectionItem],
        request: InspectionRequest
    ):
        """バッチ処理を実行"""
        try:
            # バッチステータスを更新
            await inspection_repository.update_batch_status(batch_id, "running")
            
            # デバイスモードを設定
            if request.device_mode:
                self.device_mode = request.device_mode
            
            # GPU負荷率を設定
            if request.gpu_load_rate is not None:
                self._gpu_load_manager.set_load_rate(request.gpu_load_rate)
            
            # バッチサイズと並列数を取得
            batch_size = self._gpu_load_manager.get_batch_size(self.device_mode)
            parallel_processes = self._gpu_load_manager.get_parallel_processes(self.device_mode)
            
            # 結果を格納
            results = []
            
            # バッチ処理
            for i in range(0, len(items), batch_size):
                batch_items = items[i:i + batch_size]
                
                # 並列処理
                batch_results = await asyncio.gather(
                    *[self._inspect_single_item(
                        item,
                        batch_id,
                        request.confidence_threshold,
                        request.max_detections
                    ) for item in batch_items]
                )
                
                # 結果を保存
                await inspection_repository.save_inspection_results_batch(batch_results)
                results.extend(batch_results)
                
                # 進捗を更新
                processed = len([r for r in batch_results if r.error_message is None])
                detected = len([r for r in batch_results if r.detected])
                failed = len([r for r in batch_results if r.error_message is not None])
                
                await inspection_repository.update_batch_progress(
                    batch_id, processed, detected, failed
                )
                
                # メモリ上のステータスも更新
                if batch_id in self.active_batches:
                    status = self.active_batches[batch_id]
                    status.processed_items += len(batch_results)
                    status.detected_items += detected
                    status.failed_items += failed
                    status.progress = (status.processed_items / status.total_items) * 100
            
            # バッチを完了
            await inspection_repository.complete_batch(batch_id)
            
            # ステータスを更新
            if batch_id in self.active_batches:
                self.active_batches[batch_id].status = "completed"
                self.active_batches[batch_id].end_time = datetime.utcnow()
            
            # 統計を更新
            await inspection_repository.update_statistics()
            
            logger.info(f"Batch {batch_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            await inspection_repository.update_batch_status(
                batch_id, "failed", str(e)
            )
            if batch_id in self.active_batches:
                self.active_batches[batch_id].status = "failed"
                self.active_batches[batch_id].error_message = str(e)
    
    async def _inspect_single_item(
        self,
        item: InspectionItem,
        batch_id: str,
        confidence_threshold: float,
        max_detections: int
    ) -> InspectionResult:
        """単一アイテムの検査"""
        start_time = time.time()
        
        try:
            # 画像を読み込み
            if not os.path.exists(item.image_path):
                raise FileNotFoundError(f"Image not found: {item.image_path}")
                
            image = cv2.imread(item.image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {item.image_path}")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 検出実行
            detections = self.detection_engine.detect(
                image,
                confidence_threshold=confidence_threshold,
                max_detections=max_detections
            )
            
            # 結果を整理
            result = InspectionResult(
                batch_id=batch_id,
                item_id=item.item_id,
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
            logger.error(f"Error processing {item.image_path}: {e}")
            
            # エラー結果を返す
            return InspectionResult(
                batch_id=batch_id,
                item_id=item.item_id,
                seller_id=item.seller_id,
                asin=item.asin,
                user_id=item.user_id,
                image_path=item.image_path,
                detected=False,
                detections=[],
                confidence_scores=[],
                labels=[],
                model_used=self.detection_engine.current_model_name,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def get_batch_status(self, batch_id: str) -> Optional[InspectionStatus]:
        """バッチステータスを取得"""
        # メモリから取得
        if batch_id in self.active_batches:
            return self.active_batches[batch_id]
        
        # DBから取得
        db_status = await inspection_repository.get_batch_status(batch_id)
        if db_status:
            return InspectionStatus(
                batch_id=db_status['batch_id'],
                status=db_status['status'],
                total_items=db_status['total_items'],
                processed_items=db_status['processed_items'],
                detected_items=db_status.get('detected_items', 0),
                failed_items=db_status.get('failed_items', 0),
                progress=float(db_status.get('progress', 0)),
                start_time=db_status['start_time'],
                end_time=db_status.get('end_time'),
                mode=InspectionMode(db_status['mode']),
                seller_ids=[db_status['seller_id']] if db_status.get('seller_id') else [],
                base_path=db_status.get('base_path'),
                error_message=db_status.get('error_message')
            )
        
        return None
    
    async def get_active_batches(self) -> List[InspectionStatus]:
        """アクティブなバッチ一覧を取得"""
        # DBから取得
        db_batches = await inspection_repository.get_active_batches()
        
        statuses = []
        for batch in db_batches:
            status = InspectionStatus(
                batch_id=batch['batch_id'],
                status=batch['status'],
                total_items=batch['total_items'],
                processed_items=batch['processed_items'],
                detected_items=batch.get('detected_items', 0),
                failed_items=batch.get('failed_items', 0),
                progress=float(batch.get('progress_percentage', 0)),
                start_time=batch['start_time'],
                end_time=batch.get('end_time'),
                mode=InspectionMode(batch['mode']),
                seller_ids=[batch['seller_id']] if batch.get('seller_id') else [],
                base_path=batch.get('base_path'),
                error_message=batch.get('error_message')
            )
            statuses.append(status)
        
        return statuses
    
    async def get_batch_results(
        self,
        batch_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[InspectionResult]:
        """バッチの検査結果を取得"""
        db_results = await inspection_repository.get_batch_results(
            batch_id, limit, offset
        )
        
        results = []
        for row in db_results:
            result = InspectionResult(
                batch_id=batch_id,
                item_id=row['item_id'],
                seller_id=row['seller_id'],
                asin=row['asin'],
                user_id=row['user_id'],
                image_path=row['image_path'],
                detected=row['detected'],
                detections=row['detections'],
                confidence_scores=row['confidence_scores'],
                labels=row['labels'],
                processing_time=float(row['processing_time']) if row['processing_time'] else 0,
                error_message=row.get('error_message')
            )
            results.append(result)
        
        return results
    
    async def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        return await inspection_repository.get_statistics()


# シングルトンインスタンス
_inspection_engine_v2 = None

def get_inspection_engine_v2() -> InspectionEngineV2:
    """検査エンジンのシングルトンインスタンスを取得"""
    global _inspection_engine_v2
    if _inspection_engine_v2 is None:
        _inspection_engine_v2 = InspectionEngineV2()
    return _inspection_engine_v2