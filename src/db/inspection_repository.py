"""
画像検査データベースリポジトリ
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import asyncpg

from src.db.connection import db_connection
from src.models.inspection_schemas import (
    InspectionRequest, InspectionStatus, InspectionResult,
    InspectionBatchResult, InspectionMode
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InspectionRepository:
    """画像検査データベースリポジトリ"""
    
    async def create_batch(
        self,
        request: InspectionRequest,
        total_items: int = 0
    ) -> str:
        """新しい検査バッチを作成"""
        batch_id = str(uuid.uuid4())
        
        query = """
        INSERT INTO inspection_batches (
            batch_id, status, mode, model_name, 
            confidence_threshold, max_detections,
            total_items, seller_id, base_path,
            include_subdirs
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING batch_id
        """
        
        async with db_connection.get_async_connection() as conn:
            result = await conn.fetchval(
                query,
                batch_id,
                'pending',
                request.mode.value,
                request.model_name,
                request.confidence_threshold,
                request.max_detections,
                total_items,
                request.sellers[0] if request.sellers else None,
                request.base_path,
                request.include_subdirs
            )
            
        logger.info(f"Created inspection batch: {result}")
        return result
    
    async def update_batch_status(
        self,
        batch_id: str,
        status: str,
        error_message: Optional[str] = None
    ):
        """バッチステータスを更新"""
        query = """
        UPDATE inspection_batches 
        SET status = $1, error_message = $2
        WHERE batch_id = $3
        """
        
        async with db_connection.get_async_connection() as conn:
            await conn.execute(query, status, error_message, batch_id)
    
    async def update_batch_progress(
        self,
        batch_id: str,
        processed: int = 0,
        detected: int = 0,
        failed: int = 0
    ):
        """バッチ進捗を更新"""
        query = """
        UPDATE inspection_batches 
        SET 
            processed_items = processed_items + $1,
            detected_items = detected_items + $2,
            failed_items = failed_items + $3
        WHERE batch_id = $4
        """
        
        async with db_connection.get_async_connection() as conn:
            await conn.execute(query, processed, detected, failed, batch_id)
    
    async def complete_batch(self, batch_id: str):
        """バッチを完了"""
        query = """
        UPDATE inspection_batches 
        SET 
            status = 'completed',
            end_time = CURRENT_TIMESTAMP,
            processing_time_seconds = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - start_time)),
            average_confidence = (
                SELECT AVG(confidence)
                FROM inspection_results ir
                CROSS JOIN LATERAL jsonb_array_elements(ir.detections) AS d(detection)
                WHERE ir.batch_id = $1
                AND (d.detection->>'confidence')::float > 0
            )
        WHERE batch_id = $1
        """
        
        async with db_connection.get_async_connection() as conn:
            await conn.execute(query, batch_id)
    
    async def save_inspection_result(self, result: InspectionResult):
        """検査結果をinventoryテーブルに保存（更新）"""
        query = """
        UPDATE inventory 
        SET 
            model_used = $3,
            last_processed_at = CURRENT_TIMESTAMP,
            detections = $4::jsonb,
            confidence_scores = $5,
            labels = $6,
            detected = $7,
            batch_id = $8
        WHERE seller_id = $1 AND asin = $2
        """
        
        # 検出結果をJSON形式に変換
        detections_json = json.dumps(result.detections) if result.detections else '[]'
        
        async with db_connection.get_async_connection() as conn:
            await conn.execute(
                query,
                result.seller_id,
                result.asin,
                result.model_used,
                detections_json,
                result.confidence_scores or [],
                result.labels or [],
                result.detected,
                result.batch_id
            )
    
    async def save_inspection_results_batch(self, results: List[InspectionResult]):
        """複数の検査結果を一括保存（inventoryテーブルを更新）"""
        if not results:
            return
        
        # バッチ更新用のデータを準備
        data = []
        for result in results:
            # seller_idとasinがある結果のみ処理
            if result.seller_id and result.asin:
                detections_json = json.dumps(result.detections) if result.detections else '[]'
                data.append((
                    result.seller_id,
                    result.asin,
                    result.model_used,
                    detections_json,
                    result.confidence_scores or [],
                    result.labels or [],
                    result.detected,
                    result.batch_id
                ))
        
        if not data:
            return
        
        query = """
        UPDATE inventory 
        SET 
            model_used = $3,
            last_processed_at = CURRENT_TIMESTAMP,
            detections = $4::jsonb,
            confidence_scores = $5,
            labels = $6,
            detected = $7,
            batch_id = $8
        WHERE seller_id = $1 AND asin = $2
        """
        
        async with db_connection.get_async_connection() as conn:
            await conn.executemany(query, data)
    
    async def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """バッチステータスを取得"""
        query = """
        SELECT 
            batch_id, status, mode, model_name,
            total_items, processed_items, detected_items, failed_items,
            CASE 
                WHEN total_items > 0 THEN 
                    ROUND((processed_items::DECIMAL / total_items) * 100, 2)
                ELSE 0
            END as progress,
            start_time, end_time, error_message,
            seller_id, base_path
        FROM inspection_batches
        WHERE batch_id = $1
        """
        
        async with db_connection.get_async_connection() as conn:
            row = await conn.fetchrow(query, batch_id)
            
        if row:
            return dict(row)
        return None
    
    async def get_active_batches(self) -> List[Dict[str, Any]]:
        """アクティブなバッチ一覧を取得"""
        query = """
        SELECT 
            batch_id, status, mode, model_name,
            total_items, processed_items, detected_items, failed_items,
            CASE 
                WHEN total_items > 0 THEN 
                    ROUND((processed_items::DECIMAL / total_items) * 100, 2)
                ELSE 0
            END as progress_percentage,
            start_time, end_time, 
            EXTRACT(EPOCH FROM (COALESCE(end_time, CURRENT_TIMESTAMP) - start_time)) as elapsed_seconds,
            seller_id, base_path, error_message
        FROM inspection_batches
        WHERE status IN ('pending', 'running')
        ORDER BY start_time DESC
        LIMIT 50
        """
        
        async with db_connection.get_async_connection() as conn:
            rows = await conn.fetch(query)
            
        return [dict(row) for row in rows]
    
    async def get_batch_results(
        self,
        batch_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """バッチの検査結果をinventoryテーブルから取得"""
        query = """
        SELECT 
            id as item_id,
            seller_id,
            asin,
            sku,
            image_url as image_path,
            detected,
            model_used,
            detections,
            confidence_scores,
            labels,
            last_processed_at,
            ARRAY_LENGTH(labels, 1) as detection_count
        FROM inventory
        WHERE batch_id = $1
        ORDER BY last_processed_at DESC
        LIMIT $2 OFFSET $3
        """
        
        async with db_connection.get_async_connection() as conn:
            rows = await conn.fetch(query, batch_id, limit, offset)
            
        return [dict(row) for row in rows]
    
    async def get_items_from_inventory(
        self,
        seller_id: str,
        limit: Optional[int] = None,
        only_uninspected: bool = True,
        model_used: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """inventoryテーブルからセラーIDに基づいてアイテムを取得"""
        query = """
        SELECT 
            id,
            seller_id,
            asin,
            sku,
            image_url,
            has_image_url,
            detected,
            last_processed_at,
            model_used
        FROM inventory
        WHERE seller_id = $1
        AND has_image_url = true
        """
        
        # 未検査のみフィルタ
        if only_uninspected and model_used:
            query += f" AND (model_used IS NULL OR model_used != '{model_used}')"
        
        query += " ORDER BY last_processed_at ASC NULLS FIRST"
        
        if limit:
            query += f" LIMIT {limit}"
        
        async with db_connection.get_async_connection() as conn:
            rows = await conn.fetch(query, seller_id)
            
        return [dict(row) for row in rows]
    
    async def update_statistics(self):
        """統計情報を更新"""
        query = """
        INSERT INTO inspection_statistics (date)
        VALUES (CURRENT_DATE)
        ON CONFLICT (date) DO UPDATE
        SET 
            total_inspections = (
                SELECT COUNT(*) FROM inventory
                WHERE DATE(last_processed_at) = CURRENT_DATE
            ),
            total_detections = (
                SELECT COUNT(*) FROM inventory
                WHERE DATE(last_processed_at) = CURRENT_DATE
                AND detected = true
            ),
            unique_sellers = (
                SELECT COUNT(DISTINCT seller_id) FROM inventory
                WHERE DATE(last_processed_at) = CURRENT_DATE
            )
        """
        
        async with db_connection.get_async_connection() as conn:
            await conn.execute(query)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        try:
            # シンプルな統計クエリに変更
            query = """
            SELECT 
                COUNT(*) as total_inspections,
                COUNT(CASE WHEN detected THEN 1 END) as total_detections,
                COUNT(CASE WHEN DATE(last_processed_at) = CURRENT_DATE THEN 1 END) as inspections_today,
                COUNT(CASE WHEN last_processed_at >= CURRENT_DATE - INTERVAL '7 days' THEN 1 END) as inspections_week
            FROM inventory
            WHERE batch_id IS NOT NULL
            """
        
            async with db_connection.get_async_connection() as conn:
                row = await conn.fetchrow(query)
                
            if row:
                result = dict(row)
                # 検出率を計算
                total = result.get('total_inspections', 0)
                detected = result.get('total_detections', 0)
                result['detection_rate'] = (detected / total) if total > 0 else 0
                result['average_processing_time'] = 0.1  # TODO: 実装
                result['inspections_this_week'] = result.get('inspections_week', 0)
                result['top_detected_brands'] = []  # TODO: 実装
                result['by_seller'] = []  # TODO: 実装
                return result
            
            return {
                'total_inspections': 0,
                'total_detections': 0,
                'detection_rate': 0,
                'average_processing_time': 0,
                'inspections_today': 0,
                'inspections_this_week': 0,
                'top_detected_brands': [],
                'by_seller': []
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            # エラー時はデフォルト値を返す
            return {
                'total_inspections': 0,
                'total_detections': 0,
                'detection_rate': 0,
                'average_processing_time': 0,
                'inspections_today': 0,
                'inspections_this_week': 0,
                'top_detected_brands': [],
                'by_seller': []
            }


# シングルトンインスタンス
inspection_repository = InspectionRepository()