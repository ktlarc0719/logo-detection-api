"""
画像検査APIエンドポイント V2
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List

from src.models.inspection_schemas_v2 import (
    InspectionRequest, InspectionStatus, InspectionDashboard
)
from src.core.inspection_engine_v2 import get_inspection_engine_v2
from src.core.detection_engine import get_detection_engine
from src.core.inspection_queue import get_inspection_queue, QueueStatus
from src.utils.logger import get_logger
import torch

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/inspection/start",
    summary="画像検査を開始",
    description="指定された条件で画像検査バッチを開始します"
)
async def start_inspection(request: InspectionRequest) -> dict:
    """
    画像検査を開始
    
    - **mode**: 実行モード（seller: セラーID指定, path: 絶対パス指定）
    - **model_name**: 使用するモデル名
    - **sellers**: セラーIDリスト（seller mode時）
    - **base_path**: 基準パス（path mode時）
    - **include_subdirs**: サブディレクトリを含むか（path mode時）
    - **max_items**: 処理上限数（未指定時は全件）
    """
    try:
        logger.info(f"Starting inspection: mode={request.mode}, model={request.model_name}")
        
        # モード別の検証
        if request.mode == "seller":
            if not request.sellers:
                raise HTTPException(
                    status_code=400,
                    detail="sellers list is required for seller mode"
                )
        elif request.mode == "path":
            if not request.base_path:
                raise HTTPException(
                    status_code=400,
                    detail="base_path is required for path mode"
                )
        
        # モデルを切り替え
        detection_engine = get_detection_engine()
        if not detection_engine.switch_model(request.model_name):
            raise HTTPException(
                status_code=400,
                detail=f"Failed to switch to model: {request.model_name}"
            )
        
        # キューに追加
        queue = get_inspection_queue()
        queue_id = await queue.add_to_queue(request)
        
        return {
            "success": True,
            "queue_id": queue_id,
            "message": "Inspection request added to queue"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start inspection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/inspection/status/{batch_id}",
    response_model=InspectionStatus,
    summary="検査ステータスを取得",
    description="指定されたバッチIDの検査ステータスを取得します"
)
async def get_inspection_status(batch_id: str) -> InspectionStatus:
    """バッチIDを指定して検査ステータスを取得"""
    inspection_engine = get_inspection_engine_v2()
    status = await inspection_engine.get_batch_status(batch_id)
    
    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"Batch {batch_id} not found"
        )
    
    return status


@router.get(
    "/inspection/active",
    response_model=List[InspectionStatus],
    summary="アクティブな検査一覧を取得",
    description="現在実行中または待機中の検査一覧を取得します"
)
async def get_active_inspections() -> List[InspectionStatus]:
    """アクティブな検査バッチの一覧を取得"""
    try:
        inspection_engine = get_inspection_engine_v2()
        statuses = await inspection_engine.get_active_batches()
        return statuses
    except Exception as e:
        logger.error(f"Failed to get active inspections: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/inspection/results/{batch_id}",
    summary="検査結果を取得",
    description="指定されたバッチIDの検査結果を取得します"
)
async def get_inspection_results(
    batch_id: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
) -> dict:
    """バッチIDを指定して検査結果を取得"""
    inspection_engine = get_inspection_engine_v2()
    results = await inspection_engine.get_batch_results(batch_id, limit, offset)
    
    return {
        "batch_id": batch_id,
        "results": results,
        "count": len(results),
        "limit": limit,
        "offset": offset
    }


@router.get(
    "/inspection/queue/status",
    summary="キューステータスを取得",
    description="検査キューの現在の状態を取得します"
)
async def get_queue_status() -> dict:
    """キューの状態を取得"""
    queue = get_inspection_queue()
    return await queue.get_queue_status()


@router.get(
    "/inspection/queue/{queue_id}",
    summary="キューアイテムのステータスを取得",
    description="指定されたキューIDのアイテムステータスを取得します"
)
async def get_queue_item_status(queue_id: str) -> dict:
    """キューアイテムのステータスを取得"""
    queue = get_inspection_queue()
    item = await queue.get_item_status(queue_id)
    
    if not item:
        raise HTTPException(
            status_code=404,
            detail=f"Queue item {queue_id} not found"
        )
    
    return {
        "queue_id": item.queue_id,
        "batch_id": item.batch_id,
        "status": item.status,
        "position": item.position,
        "created_at": item.created_at.isoformat(),
        "started_at": item.started_at.isoformat() if item.started_at else None,
        "completed_at": item.completed_at.isoformat() if item.completed_at else None,
        "error_message": item.error_message
    }


@router.delete(
    "/inspection/queue/{queue_id}",
    summary="キューアイテムをキャンセル",
    description="指定されたキューIDのアイテムをキャンセルします"
)
async def cancel_queue_item(queue_id: str) -> dict:
    """キューアイテムをキャンセル"""
    queue = get_inspection_queue()
    success = await queue.cancel_item(queue_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Queue item {queue_id} not found or cannot be cancelled"
        )
    
    return {
        "success": True,
        "message": f"Queue item {queue_id} cancelled"
    }


@router.get(
    "/inspection/models",
    summary="利用可能なモデル一覧を取得",
    description="検査で使用可能なモデルの一覧を取得します"
)
async def get_available_models() -> dict:
    """利用可能なモデル一覧を取得"""
    try:
        detection_engine = get_detection_engine()
        models_dict = detection_engine.get_available_models()
        current_model = detection_engine.current_model_name
        
        # models_dictは{model_name: model_info}の形式なので、リストに変換
        models_list = []
        for name, info in models_dict.items():
            models_list.append({
                "name": name,
                "loaded": info.get("loaded", False),
                "is_current": info.get("is_current", False),
                "description": info.get("description", ""),
                "type": info.get("source", "general")
            })
        
        return {
            "models": models_list,
            "current_model": current_model
        }
    except Exception as e:
        logger.error(f"Failed to get available models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/inspection/device-info",
    summary="デバイス情報を取得",
    description="GPU/CPUのデバイス情報を取得します"
)
async def get_device_info() -> dict:
    """デバイス情報を取得"""
    gpu_available = torch.cuda.is_available()
    gpu_info = {}
    
    if gpu_available:
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_total": torch.cuda.get_device_properties(0).total_memory,
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_cached": torch.cuda.memory_reserved(0),
            "compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
        }
    
    return {
        "gpu_status": {
            "available": gpu_available,
            "info": gpu_info if gpu_available else None
        },
        "current_device_mode": "gpu" if gpu_available else "cpu",
        "cpu_info": {
            "cores": torch.get_num_threads()
        }
    }


@router.get(
    "/inspection/dashboard",
    response_model=InspectionDashboard,
    summary="ダッシュボード情報を取得",
    description="検査管理UIで使用するダッシュボード情報を取得します"
)
async def get_inspection_dashboard() -> InspectionDashboard:
    """ダッシュボード用の統合情報を取得"""
    try:
        inspection_engine = get_inspection_engine_v2()
        detection_engine = get_detection_engine()
        
        # アクティブバッチ
        active_batches = await inspection_engine.get_active_batches()
        
        # 利用可能なモデル（Dict形式をListに変換）
        models_dict = detection_engine.get_available_models()
        available_models = [
            {"name": name, **info}
            for name, info in models_dict.items()
        ]
        
        # GPU状態
        gpu_status = {
            "available": torch.cuda.is_available(),
            "load_rate": 0.8  # デフォルト値
        }
        
        # 統計情報
        statistics = await inspection_engine.get_statistics()
        
        return InspectionDashboard(
            active_batches=active_batches,
            available_models=available_models,
            gpu_status=gpu_status,
            recent_results=[],  # TODO: 実装
            statistics=statistics
        )
    except Exception as e:
        logger.error(f"Failed to get inspection dashboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))