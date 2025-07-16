"""
画像検査APIエンドポイント
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List

from src.models.inspection_schemas_v2 import (
    InspectionRequest, InspectionStatus, InspectionDashboard
)
from src.core.inspection_engine_v2 import get_inspection_engine_v2
from src.core.detection_engine import get_detection_engine
from src.utils.logger import get_logger

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
    
    - **mode**: 実行モード（individual: 個別指定, bulk: 全件実行）
    - **model_name**: 使用するモデル名
    - **gpu_load_rate**: GPU負荷率（0.1-1.0）
    - **seller_id**: セラーID（個別指定モード時）
    - **user_id**: ユーザーID（個別指定モード時）
    - **max_items**: 処理上限数
    - **process_all**: 全件処理フラグ（全件実行モード時）
    """
    try:
        logger.info(f"Starting inspection: mode={request.mode}, model={request.model_name}")
        
        # 個別指定モードの検証
        if request.mode == "individual":
            if not request.seller_id:
                raise HTTPException(
                    status_code=400,
                    detail="seller_id is required for individual mode"
                )
        
        # 検査エンジンを取得
        engine = get_inspection_engine()
        
        # バッチ処理を開始
        batch_id = await engine.process_inspection_batch(request)
        
        return {
            "success": True,
            "batch_id": batch_id,
            "message": "Inspection batch started successfully"
        }
        
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
    """特定のバッチの検査ステータスを取得"""
    engine = get_inspection_engine()
    status = engine.get_batch_status(batch_id)
    
    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"Batch {batch_id} not found"
        )
    
    return status


@router.get(
    "/inspection/status",
    response_model=List[InspectionStatus],
    summary="全検査ステータスを取得",
    description="実行中の全ての検査バッチのステータスを取得します"
)
async def get_all_inspection_statuses() -> List[InspectionStatus]:
    """全ての検査ステータスを取得"""
    engine = get_inspection_engine()
    return engine.get_all_batch_statuses()


@router.post(
    "/inspection/cancel/{batch_id}",
    summary="検査をキャンセル",
    description="指定されたバッチIDの検査をキャンセルします"
)
async def cancel_inspection(batch_id: str) -> dict:
    """検査をキャンセル"""
    engine = get_inspection_engine()
    success = engine.cancel_batch(batch_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Batch {batch_id} not found"
        )
    
    return {
        "success": True,
        "message": f"Batch {batch_id} cancelled"
    }


@router.get(
    "/inspection/dashboard",
    response_model=InspectionDashboard,
    summary="管理ダッシュボードデータを取得",
    description="管理UI用のダッシュボードデータを取得します"
)
async def get_dashboard() -> InspectionDashboard:
    """管理ダッシュボードデータを取得"""
    try:
        inspection_engine = get_inspection_engine_v2()
        detection_engine = get_detection_engine()
        
        # アクティブなバッチ
        active_batches = inspection_engine.get_all_batch_statuses()
        
        # 利用可能なモデル
        available_models = []
        for model_name, model_info in detection_engine.get_available_models().items():
            available_models.append({
                "name": model_name,
                "loaded": model_info.get("loaded", False),
                "is_current": model_info.get("is_current", False),
                "path": model_info.get("path", ""),
                "confidence_threshold": model_info.get("confidence_threshold", 0.5)
            })
        
        # GPU状態
        gpu_status = inspection_engine._gpu_load_manager.get_gpu_status()
        
        # 統計情報
        statistics = await get_inspection_statistics()
        
        # 最近の結果（ダミー）
        recent_results = []  # TODO: 実装
        
        return InspectionDashboard(
            active_batches=active_batches,
            available_models=available_models,
            gpu_status=gpu_status,
            recent_results=recent_results,
            statistics=statistics
        )
        
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/inspection/models",
    summary="利用可能なモデル一覧を取得",
    description="画像検査に使用可能なモデルの一覧を取得します"
)
async def get_available_models() -> dict:
    """利用可能なモデル一覧を取得"""
    detection_engine = get_detection_engine()
    models = detection_engine.get_available_models()
    
    return {
        "models": [
            {
                "name": name,
                "loaded": info.get("loaded", False),
                "is_current": info.get("is_current", False),
                "classes": info.get("classes", {}),
                "confidence_threshold": info.get("confidence_threshold", 0.5)
            }
            for name, info in models.items()
        ]
    }


@router.post(
    "/inspection/batch",
    summary="検査バッチを開始（新）",
    description="画像検査バッチを開始します（/inspection/startの代替）"
)
async def start_inspection_batch(request: InspectionRequest) -> dict:
    """検査バッチを開始（startと同じ機能）"""
    return await start_inspection(request)


@router.get(
    "/inspection/batch/{batch_id}/status",
    response_model=InspectionStatus,
    summary="バッチステータスを取得（新）",
    description="指定されたバッチIDのステータスを取得します"
)
async def get_batch_status(batch_id: str) -> InspectionStatus:
    """バッチステータスを取得"""
    return await get_inspection_status(batch_id)


@router.get(
    "/inspection/batch/{batch_id}/results",
    summary="バッチ結果を取得",
    description="指定されたバッチIDの検査結果を取得します"
)
async def get_batch_results(batch_id: str) -> dict:
    """バッチ結果を取得"""
    try:
        engine = get_inspection_engine()
        
        # 実際の結果を取得
        batch_result = engine.get_batch_results(batch_id)
        
        if not batch_result:
            # ステータスを確認
            status = engine.get_batch_status(batch_id)
            if not status:
                raise HTTPException(
                    status_code=404,
                    detail=f"Batch {batch_id} not found"
                )
            elif status.status == "running":
                raise HTTPException(
                    status_code=202,
                    detail=f"Batch {batch_id} is still running"
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Results for batch {batch_id} not found"
                )
        
        # 結果をシリアライズ可能な形式に変換
        results = []
        for result in batch_result.results:
            results.append({
                "item_id": result.item_id,
                "seller_id": result.seller_id,
                "asin": result.asin,
                "image_path": result.image_path,
                "detected": result.detected,
                "labels": result.labels,
                "confidence_scores": result.confidence_scores,
                "detections": result.detections,  # バウンディングボックス情報を含む
                "model_used": result.model_used,
                "processing_time": result.processing_time,
                "error": result.error
            })
        
        return {
            "batch_id": batch_result.batch_id,
            "total_items": batch_result.total_items,
            "successful_items": batch_result.successful_items,
            "failed_items": batch_result.failed_items,
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get batch results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/inspection/device-info",
    summary="デバイス情報を取得",
    description="CPU/GPU の利用可能状況を取得します"
)
async def get_device_info() -> dict:
    """デバイス情報を取得"""
    import torch
    
    inspection_engine = get_inspection_engine_v2()
    gpu_status = inspection_engine._gpu_load_manager.get_gpu_status()
    
    return {
        "cpu_available": True,
        "gpu_status": gpu_status,
        "current_device_mode": inspection_engine.device_mode,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }


@router.get(
    "/inspection/image",
    summary="画像を取得",
    description="指定されたパスの画像ファイルを取得します"
)
async def get_inspection_image(image_path: str = Query(..., description="画像ファイルのパス")):
    """画像ファイルを取得"""
    import os
    from fastapi.responses import FileResponse
    
    # セキュリティチェック: パストラバーサル攻撃を防ぐ
    if ".." in image_path:
        raise HTTPException(
            status_code=400,
            detail="Invalid image path"
        )
    
    # フルパスが渡された場合の処理
    if image_path.startswith("/mnt/"):
        full_path = image_path
    else:
        # 相対パスの場合はベースディレクトリを付ける
        full_path = os.path.join("/mnt/c/03_amazon_images", image_path)
    
    # ファイルの存在確認
    if not os.path.exists(full_path):
        raise HTTPException(
            status_code=404,
            detail=f"Image not found: {image_path}"
        )
    
    # ファイルが画像かどうかチェック
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    file_ext = os.path.splitext(full_path)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="File is not an image"
        )
    
    return FileResponse(full_path, media_type=f"image/{file_ext[1:]}")