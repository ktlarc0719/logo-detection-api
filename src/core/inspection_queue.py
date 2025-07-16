"""
検査キューイングシステム
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import uuid
from enum import Enum

from src.models.inspection_schemas_v2 import InspectionRequest
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QueueStatus(str, Enum):
    """キューステータス"""
    PENDING = "pending"      # 待機中
    PROCESSING = "processing" # 処理中
    COMPLETED = "completed"  # 完了
    FAILED = "failed"       # 失敗
    CANCELLED = "cancelled" # キャンセル


@dataclass
class QueueItem:
    """キューアイテム"""
    queue_id: str
    batch_id: Optional[str]
    request: InspectionRequest
    status: QueueStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    position: int = 0  # キュー内の位置


class InspectionQueue:
    """検査キューマネージャー"""
    
    def __init__(self):
        self._queue: List[QueueItem] = []
        self._processing_item: Optional[QueueItem] = None
        self._lock = asyncio.Lock()
        self._processor_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
    async def start(self):
        """キュー処理を開始"""
        if self._processor_task is None:
            self._processor_task = asyncio.create_task(self._process_queue())
            logger.info("Inspection queue processor started")
            
    async def stop(self):
        """キュー処理を停止"""
        self._shutdown = True
        if self._processor_task:
            await self._processor_task
            self._processor_task = None
        logger.info("Inspection queue processor stopped")
        
    async def add_to_queue(self, request: InspectionRequest) -> str:
        """キューに追加"""
        async with self._lock:
            queue_id = str(uuid.uuid4())
            item = QueueItem(
                queue_id=queue_id,
                batch_id=None,
                request=request,
                status=QueueStatus.PENDING,
                created_at=datetime.utcnow(),
                position=len(self._queue) + 1
            )
            self._queue.append(item)
            logger.info(f"Added item to queue: {queue_id}, position: {item.position}")
            return queue_id
            
    async def get_queue_status(self) -> Dict[str, Any]:
        """キューの状態を取得"""
        async with self._lock:
            return {
                "queue_length": len(self._queue),
                "processing": self._processing_item is not None,
                "processing_item": {
                    "queue_id": self._processing_item.queue_id,
                    "batch_id": self._processing_item.batch_id,
                    "started_at": self._processing_item.started_at.isoformat() if self._processing_item.started_at else None
                } if self._processing_item else None,
                "pending_items": [
                    {
                        "queue_id": item.queue_id,
                        "position": idx + 1,
                        "created_at": item.created_at.isoformat(),
                        "seller_id": item.request.sellers[0] if item.request.sellers else None
                    }
                    for idx, item in enumerate(self._queue)
                ]
            }
            
    async def get_item_status(self, queue_id: str) -> Optional[QueueItem]:
        """特定アイテムのステータスを取得"""
        async with self._lock:
            # 処理中のアイテムをチェック
            if self._processing_item and self._processing_item.queue_id == queue_id:
                return self._processing_item
                
            # キュー内のアイテムをチェック
            for idx, item in enumerate(self._queue):
                if item.queue_id == queue_id:
                    item.position = idx + 1
                    return item
                    
            return None
            
    async def cancel_item(self, queue_id: str) -> bool:
        """キューアイテムをキャンセル"""
        async with self._lock:
            # キュー内のアイテムのみキャンセル可能
            for idx, item in enumerate(self._queue):
                if item.queue_id == queue_id:
                    item.status = QueueStatus.CANCELLED
                    self._queue.pop(idx)
                    logger.info(f"Cancelled queue item: {queue_id}")
                    return True
            return False
            
    async def _process_queue(self):
        """キューを処理"""
        while not self._shutdown:
            try:
                async with self._lock:
                    if self._queue and self._processing_item is None:
                        # キューから次のアイテムを取得
                        self._processing_item = self._queue.pop(0)
                        self._processing_item.status = QueueStatus.PROCESSING
                        self._processing_item.started_at = datetime.utcnow()
                        
                        # キュー内の残りのアイテムの位置を更新
                        for idx, item in enumerate(self._queue):
                            item.position = idx + 1
                            
                if self._processing_item:
                    try:
                        # 実際の検査処理を実行
                        from src.core.inspection_engine_v2 import get_inspection_engine_v2
                        engine = get_inspection_engine_v2()
                        
                        batch_id = await engine.start_inspection(self._processing_item.request)
                        self._processing_item.batch_id = batch_id
                        
                        # 検査完了を待つ
                        while True:
                            status = await engine.get_batch_status(batch_id)
                            if status and status.status in ["completed", "failed", "cancelled"]:
                                break
                            await asyncio.sleep(1)
                            
                        self._processing_item.status = QueueStatus.COMPLETED
                        self._processing_item.completed_at = datetime.utcnow()
                        logger.info(f"Completed queue item: {self._processing_item.queue_id}")
                        
                    except Exception as e:
                        self._processing_item.status = QueueStatus.FAILED
                        self._processing_item.error_message = str(e)
                        self._processing_item.completed_at = datetime.utcnow()
                        logger.error(f"Failed to process queue item: {e}")
                        
                    finally:
                        async with self._lock:
                            self._processing_item = None
                            
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
                
            # 少し待機
            await asyncio.sleep(0.5)


# シングルトンインスタンス
_inspection_queue: Optional[InspectionQueue] = None


def get_inspection_queue() -> InspectionQueue:
    """検査キューのシングルトンインスタンスを取得"""
    global _inspection_queue
    if _inspection_queue is None:
        _inspection_queue = InspectionQueue()
    return _inspection_queue