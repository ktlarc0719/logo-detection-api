"""
機械学習システムAPIエンドポイント
"""

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from typing import List, Optional
import os
import psutil
import torch

from src.models.ml_system_schemas import (
    DatasetValidationRequest, DatasetValidationResult,
    TrainingRequest, TrainingStatus,
    ModelVisualizationRequest, ModelVisualizationResult,
    ModelValidationRequest, ModelValidationResult,
    SystemStatus
)
from src.core.ml_dataset_validator import get_dataset_validator
from src.core.ml_training_engine import get_ml_training_engine
from src.core.ml_model_visualizer import get_model_visualizer
from src.core.ml_model_validator import get_model_validator
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """システム全体のステータスを取得"""
    try:
        # GPU情報を取得
        gpu_available = torch.cuda.is_available()
        gpu_info = None
        
        if gpu_available:
            gpu_info = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated": torch.cuda.memory_allocated(0) / 1024**3,  # GB
                "memory_reserved": torch.cuda.memory_reserved(0) / 1024**3,  # GB
            }
        
        # CPU・メモリ使用率
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # ディスク使用率
        disk = psutil.disk_usage('/')
        disk_usage = {
            "total": disk.total / 1024**3,  # GB
            "used": disk.used / 1024**3,  # GB
            "free": disk.free / 1024**3,  # GB
            "percent": disk.percent
        }
        
        # アクティブなタスク
        training_engine = get_ml_training_engine()
        active_trainings = [status.training_id for status in training_engine.get_all_training_statuses() 
                           if status.status == "running"]
        
        # 利用可能なモデル
        available_models = training_engine.get_available_models()
        
        # 利用可能なデータセット
        available_datasets = []
        dataset_dir = "datasets"
        if os.path.exists(dataset_dir):
            for item in os.listdir(dataset_dir):
                item_path = os.path.join(dataset_dir, item)
                if os.path.isdir(item_path):
                    yaml_path = os.path.join(item_path, "dataset.yaml")
                    if os.path.exists(yaml_path):
                        available_datasets.append({
                            "name": item,
                            "path": item_path
                        })
        
        return SystemStatus(
            gpu_available=gpu_available,
            gpu_info=gpu_info,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            active_trainings=active_trainings,
            active_validations=[],  # TODO: 実装
            available_models=available_models,
            available_datasets=available_datasets
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# データセット検証エンドポイント
@router.post("/dataset/validate", response_model=DatasetValidationResult)
async def validate_dataset(request: DatasetValidationRequest):
    """データセットを検証"""
    try:
        validator = get_dataset_validator()
        result = validator.validate_dataset(request)
        return result
    except Exception as e:
        logger.error(f"Dataset validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# トレーニングエンドポイント
@router.post("/training/start", response_model=dict)
async def start_training(request: TrainingRequest):
    """トレーニングを開始"""
    try:
        training_engine = get_ml_training_engine()
        training_id = await training_engine.start_training(request)
        return {"training_id": training_id, "message": "Training started successfully"}
    except Exception as e:
        logger.error(f"Failed to start training: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/training/status/{training_id}", response_model=TrainingStatus)
async def get_training_status(training_id: str):
    """トレーニングステータスを取得"""
    training_engine = get_ml_training_engine()
    status = training_engine.get_training_status(training_id)
    
    if not status:
        raise HTTPException(status_code=404, detail=f"Training {training_id} not found")
    
    return status


@router.get("/training/status", response_model=List[TrainingStatus])
async def get_all_training_statuses(
    status: Optional[str] = Query(None, description="Filter by status")
):
    """全てのトレーニングステータスを取得"""
    training_engine = get_ml_training_engine()
    statuses = training_engine.get_all_training_statuses()
    
    if status:
        statuses = [s for s in statuses if s.status == status]
    
    return statuses


@router.post("/training/cancel/{training_id}")
async def cancel_training(training_id: str):
    """トレーニングをキャンセル"""
    training_engine = get_ml_training_engine()
    success = training_engine.cancel_training(training_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Training {training_id} not found")
    
    return {"message": f"Training {training_id} cancelled"}


# モデル可視化エンドポイント
@router.post("/model/visualize", response_model=ModelVisualizationResult)
async def visualize_model(request: ModelVisualizationRequest):
    """モデルを可視化"""
    try:
        visualizer = get_model_visualizer()
        result = visualizer.visualize_model(request)
        return result
    except Exception as e:
        logger.error(f"Model visualization failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# モデル検証エンドポイント
@router.post("/model/validate", response_model=ModelValidationResult)
async def validate_model(request: ModelValidationRequest):
    """モデルを検証"""
    try:
        validator = get_model_validator()
        result = validator.validate_model(request)
        return result
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# モデル管理エンドポイント
@router.get("/models", response_model=List[dict])
async def get_available_models():
    """利用可能なモデルを取得"""
    training_engine = get_ml_training_engine()
    models = training_engine.get_available_models()
    return models


@router.get("/models/{model_name}/info")
async def get_model_info(model_name: str):
    """モデルの詳細情報を取得"""
    # モデルを探す
    training_engine = get_ml_training_engine()
    models = training_engine.get_available_models()
    
    model_info = None
    for model in models:
        if model['name'] == model_name:
            model_info = model
            break
    
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    return model_info


# データセット管理エンドポイント
@router.get("/datasets", response_model=List[dict])
async def get_available_datasets():
    """利用可能なデータセットを取得"""
    datasets = []
    dataset_dir = "datasets"
    
    if os.path.exists(dataset_dir):
        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if os.path.isdir(item_path):
                yaml_path = os.path.join(item_path, "dataset.yaml")
                if not os.path.exists(yaml_path):
                    yaml_path = os.path.join(item_path, "data.yaml")
                
                if os.path.exists(yaml_path):
                    # 簡易的な情報収集
                    info = {
                        "name": item,
                        "path": item_path,
                        "has_train": os.path.exists(os.path.join(item_path, "train")),
                        "has_val": os.path.exists(os.path.join(item_path, "val")),
                        "has_test": os.path.exists(os.path.join(item_path, "test"))
                    }
                    datasets.append(info)
    
    return datasets


# ファイルアップロードエンドポイント（将来の拡張用）
@router.post("/upload/dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: str = Query(..., description="Name for the dataset")
):
    """データセットをアップロード（ZIP形式）"""
    # TODO: 実装
    return {"message": "Dataset upload not implemented yet"}


@router.post("/upload/model")
async def upload_model(
    file: UploadFile = File(...),
    model_name: str = Query(..., description="Name for the model")
):
    """モデルをアップロード"""
    # TODO: 実装
    return {"message": "Model upload not implemented yet"}