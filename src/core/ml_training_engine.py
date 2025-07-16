"""
機械学習トレーニングエンジン
"""

import os
import time
import shutil
import asyncio
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime
import uuid
import yaml
import torch

from ultralytics import YOLO

from src.models.ml_system_schemas import (
    TrainingRequest, TrainingStatus, TrainingMode, ModelArchitecture
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLTrainingEngine:
    """機械学習トレーニングエンジン"""
    
    def __init__(self):
        self.active_trainings: Dict[str, TrainingStatus] = {}
        self.output_base_dir = Path("ml_outputs")
        self.output_base_dir.mkdir(exist_ok=True)
        
        # モデルアーキテクチャのマッピング
        self.architecture_map = {
            ModelArchitecture.YOLOV8N: "yolov8n.pt",
            ModelArchitecture.YOLOV8S: "yolov8s.pt",
            ModelArchitecture.YOLOV8M: "yolov8m.pt",
            ModelArchitecture.YOLOV8L: "yolov8l.pt",
            ModelArchitecture.YOLOV8X: "yolov8x.pt"
        }
        
        # GPU/CPU情報をログ出力
        self._log_device_info()
    
    async def start_training(self, request: TrainingRequest) -> str:
        """トレーニングを開始"""
        training_id = str(uuid.uuid4())
        
        # 実験名を生成
        if not request.experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            request.experiment_name = f"{request.mode.value}_{request.model_architecture.value}_{timestamp}"
        
        # 出力ディレクトリを作成
        output_dir = self.output_base_dir / request.project_name / request.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ステータスを初期化
        status = TrainingStatus(
            training_id=training_id,
            status="initializing",
            mode=request.mode,
            current_epoch=0,
            total_epochs=request.epochs,
            progress=0.0,
            start_time=datetime.utcnow(),
            output_dir=str(output_dir)
        )
        
        self.active_trainings[training_id] = status
        
        # バックグラウンドでトレーニングを開始
        asyncio.create_task(self._train_async(training_id, request, output_dir))
        
        return training_id
    
    async def _train_async(self, training_id: str, request: TrainingRequest, output_dir: Path):
        """非同期でトレーニングを実行"""
        try:
            self.active_trainings[training_id].status = "running"
            
            # モデルを準備
            if request.mode == TrainingMode.FULL:
                # 新規学習
                model_path = self.architecture_map[request.model_architecture]
                model = YOLO(model_path)
                logger.info(f"Starting full training with {model_path}")
                
            elif request.mode == TrainingMode.TRANSFER:
                # 転移学習: 事前学習済みモデルを新しいデータセットで再学習
                if not request.base_model_path:
                    # デフォルトのプリトレーンドモデルを使用
                    model_path = self.architecture_map[request.model_architecture]
                    model = YOLO(model_path)
                    logger.info(f"Starting transfer learning with pretrained {model_path}")
                else:
                    # カスタムモデルから転移学習
                    if not Path(request.base_model_path).exists():
                        raise ValueError(f"Base model not found: {request.base_model_path}")
                    model = YOLO(request.base_model_path)
                    logger.info(f"Starting transfer learning from custom model: {request.base_model_path}")
                    
            elif request.mode == TrainingMode.CONTINUE:
                # 継続学習: 中断したトレーニングを再開
                if not request.base_model_path:
                    raise ValueError("Base model path is required for continue learning")
                
                if not Path(request.base_model_path).exists():
                    raise ValueError(f"Base model not found: {request.base_model_path}")
                
                # 最後のチェックポイントから再開
                model = YOLO(request.base_model_path)
                logger.info(f"Continuing training from checkpoint: {request.base_model_path}")
            
            # データセットの検証
            dataset_yaml = Path(request.dataset_path) / "dataset.yaml"
            if not dataset_yaml.exists():
                dataset_yaml = Path(request.dataset_path) / "data.yaml"
            
            if not dataset_yaml.exists():
                raise ValueError("Dataset YAML not found")
            
            # トレーニングパラメータを準備
            train_args = self._prepare_training_args(request, output_dir)
            
            # トレーニング中の進捗更新用フラグ
            self._training_in_progress = True
            
            # トレーニングを実行
            try:
                logger.info(f"Starting training with dataset: {dataset_yaml}")
                logger.info(f"Training args: {train_args}")
                results = model.train(data=str(dataset_yaml), **train_args)
            except Exception as train_error:
                logger.error(f"Training error details: {str(train_error)}")
                logger.error(f"Dataset YAML path: {dataset_yaml}")
                logger.error(f"Dataset YAML exists: {dataset_yaml.exists()}")
                if dataset_yaml.exists():
                    with open(dataset_yaml, 'r') as f:
                        logger.error(f"Dataset YAML content:\n{f.read()}")
                raise
            
            # 結果を保存
            self.active_trainings[training_id].status = "completed"
            self.active_trainings[training_id].progress = 100.0
            
            # モデルパスを更新
            weights_dir = output_dir / "weights"
            if weights_dir.exists():
                best_model = weights_dir / "best.pt"
                last_model = weights_dir / "last.pt"
                
                if best_model.exists():
                    self.active_trainings[training_id].best_model_path = str(best_model)
                if last_model.exists():
                    self.active_trainings[training_id].last_model_path = str(last_model)
            
            # トレーニング設定を保存
            self._save_training_config(training_id, request, output_dir)
            
            logger.info(f"Training {training_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Training {training_id} failed: {str(e)}")
            self.active_trainings[training_id].status = "failed"
            self.active_trainings[training_id].errors.append(str(e))
    
    def _prepare_training_args(self, request: TrainingRequest, output_dir: Path) -> Dict[str, Any]:
        """トレーニング引数を準備"""
        args = {
            'epochs': request.epochs,
            'batch': request.batch_size,
            'imgsz': request.imgsz,
            'optimizer': 'SGD',  # SGDに戻す（YOLOv8のデフォルト）
            'lr0': 0.01,  # デフォルトの学習率に固定
            'patience': request.patience,
            'save_period': request.save_period,
            'project': str(output_dir.parent),
            'name': output_dir.name,
            'exist_ok': True,
            'verbose': True,
            'plots': True,
            'save': True,
            'cache': False,
            'device': self._get_device(request.device),
            'workers': 0,  # CPU環境でのマルチプロセッシング問題を回避
            'amp': False,  # 自動混合精度を無効化（CPU環境用）
            'pretrained': True,  # 事前学習済み重みを使用
            'freeze': 0,  # すべてのレイヤーを学習可能に
            'single_cls': False,  # マルチクラス分類を有効化
            'rect': False,  # 矩形トレーニングを無効化（CPU用）
        }
        
        # デフォルトの拡張パラメータ
        default_augmentation = {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,  # 回転を無効化（安定性のため）
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }
        
        # カスタム拡張パラメータで上書き
        augmentation = {**default_augmentation, **request.augmentation_params}
        args.update(augmentation)
        
        # モード別の追加設定
        if request.mode == TrainingMode.CONTINUE:
            # 継続学習: resumeフラグを設定（中断したトレーニングを再開）
            args['resume'] = True
        elif request.mode == TrainingMode.TRANSFER:
            # 転移学習: より小さい学習率を使用（ファインチューニング）
            if 'lr0' in args and args['lr0'] > 0.001:
                args['lr0'] = args['lr0'] * 0.1  # 学習率を1/10に
                logger.info(f"Transfer learning: reduced learning rate to {args['lr0']}")
        
        return args
    
    def _save_training_config(self, training_id: str, request: TrainingRequest, output_dir: Path):
        """トレーニング設定を保存"""
        config = {
            'training_id': training_id,
            'request': request.dict(),
            'status': self.active_trainings[training_id].dict(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        config_path = output_dir / 'training_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def get_training_status(self, training_id: str) -> Optional[TrainingStatus]:
        """トレーニングステータスを取得"""
        return self.active_trainings.get(training_id)
    
    def get_all_training_statuses(self) -> List[TrainingStatus]:
        """全てのトレーニングステータスを取得"""
        return list(self.active_trainings.values())
    
    def cancel_training(self, training_id: str) -> bool:
        """トレーニングをキャンセル"""
        if training_id in self.active_trainings:
            self.active_trainings[training_id].status = "cancelled"
            # TODO: 実際のトレーニングプロセスを停止
            return True
        return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """利用可能なモデルを取得"""
        models = []
        
        # トレーニング済みモデルを検索
        for project_dir in self.output_base_dir.iterdir():
            if project_dir.is_dir():
                for exp_dir in project_dir.iterdir():
                    if exp_dir.is_dir():
                        weights_dir = exp_dir / "weights"
                        if weights_dir.exists():
                            best_model = weights_dir / "best.pt"
                            if best_model.exists():
                                # トレーニング設定を読み込み
                                config_path = exp_dir / "training_config.yaml"
                                config = {}
                                if config_path.exists():
                                    with open(config_path, 'r') as f:
                                        config = yaml.safe_load(f)
                                
                                models.append({
                                    'name': exp_dir.name,
                                    'path': str(best_model),
                                    'project': project_dir.name,
                                    'created_at': best_model.stat().st_ctime,
                                    'size': best_model.stat().st_size,
                                    'config': config
                                })
        
        # 作成日時でソート
        models.sort(key=lambda x: x['created_at'], reverse=True)
        return models
    
    def _get_device(self, requested_device: Optional[str] = None) -> str:
        """使用するデバイスを決定"""
        if requested_device:
            # ユーザーが明示的に指定した場合
            if requested_device == 'cpu':
                logger.info("Using CPU as requested")
                return 'cpu'
            elif requested_device.startswith('cuda:') or requested_device == '0':
                # GPU指定
                if torch.cuda.is_available():
                    device_id = requested_device if requested_device.startswith('cuda:') else f'cuda:{requested_device}'
                    logger.info(f"Using requested GPU: {device_id}")
                    return device_id
                else:
                    logger.warning(f"GPU {requested_device} requested but CUDA is not available. Falling back to CPU")
                    return 'cpu'
        
        # 自動検出
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name} (Total GPUs: {gpu_count})")
            return '0'  # デフォルトで最初のGPUを使用
        else:
            logger.info("No GPU detected. Using CPU for training")
            return 'cpu'
    
    def _log_device_info(self):
        """デバイス情報をログ出力"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"CUDA is available with {gpu_count} GPU(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                logger.info(f"  GPU {i}: {gpu_name} ({total_memory:.2f} GB)")
        else:
            logger.info("CUDA is not available. CPU will be used for training")
            
        # CPU情報も出力
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        logger.info(f"CPU cores available: {cpu_count}")


# シングルトンインスタンス
_ml_training_engine: Optional[MLTrainingEngine] = None


def get_ml_training_engine() -> MLTrainingEngine:
    """トレーニングエンジンのインスタンスを取得"""
    global _ml_training_engine
    if _ml_training_engine is None:
        _ml_training_engine = MLTrainingEngine()
    return _ml_training_engine