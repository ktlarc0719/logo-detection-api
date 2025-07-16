"""
モデル可視化モジュール
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from ultralytics import YOLO

from src.models.ml_system_schemas import (
    ModelVisualizationRequest, ModelVisualizationResult
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelVisualizer:
    """モデル可視化クラス"""
    
    def __init__(self):
        self.output_dir = Path("ml_outputs/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # プロットスタイルを設定
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('ggplot')
    
    def visualize_model(self, request: ModelVisualizationRequest) -> ModelVisualizationResult:
        """モデルの可視化を実行"""
        logger.info(f"Visualizing model: {request.model_path}")
        
        model_path = Path(request.model_path)
        if not model_path.exists():
            raise ValueError(f"Model not found: {model_path}")
        
        # モデルをロード
        try:
            model = YOLO(str(model_path))
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
        
        # 出力ディレクトリを作成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_dir = self.output_dir / f"{model_path.stem}_{timestamp}"
        viz_dir.mkdir(exist_ok=True)
        
        # モデル情報を取得
        model_info = self._get_model_info(model, model_path)
        
        # トレーニング結果のディレクトリを探す
        training_dir = self._find_training_dir(model_path)
        
        # クラス別メトリクスを計算
        class_metrics = {}
        overall_metrics = {}
        
        # 可視化を実行
        visualization_paths = {}
        
        if request.include_confusion_matrix and training_dir:
            cm_path = self._visualize_confusion_matrix(training_dir, viz_dir)
            if cm_path:
                visualization_paths['confusion_matrix_path'] = str(cm_path)
        
        if request.include_pr_curve and training_dir:
            pr_path = self._visualize_pr_curve(training_dir, viz_dir)
            if pr_path:
                visualization_paths['pr_curve_path'] = str(pr_path)
        
        if request.include_f1_curve and training_dir:
            f1_path = self._visualize_f1_curve(training_dir, viz_dir)
            if f1_path:
                visualization_paths['f1_curve_path'] = str(f1_path)
        
        training_history = None
        if request.include_training_history and training_dir:
            training_history = self._get_training_history(training_dir, viz_dir)
        
        if request.include_class_metrics and training_dir:
            class_metrics, overall_metrics = self._calculate_class_metrics(training_dir, model)
            self._visualize_class_metrics(class_metrics, viz_dir)
        
        # トレーニング引数を取得
        training_args = None
        if training_dir:
            args_path = training_dir / "args.yaml"
            if args_path.exists():
                with open(args_path, 'r') as f:
                    training_args = yaml.safe_load(f)
        
        return ModelVisualizationResult(
            model_path=str(model_path),
            model_info=model_info,
            class_metrics=class_metrics,
            overall_metrics=overall_metrics,
            training_history=training_history,
            training_args=training_args,
            **visualization_paths
        )
    
    def _get_model_info(self, model: YOLO, model_path: Path) -> Dict[str, Any]:
        """モデル情報を取得"""
        info = {
            'path': str(model_path),
            'size': model_path.stat().st_size / (1024 * 1024),  # MB
            'created': datetime.fromtimestamp(model_path.stat().st_ctime).isoformat(),
        }
        
        # モデルのアーキテクチャ情報
        if hasattr(model, 'model'):
            if hasattr(model.model, 'nc'):
                info['num_classes'] = model.model.nc
            if hasattr(model.model, 'names'):
                info['class_names'] = model.model.names
        
        # YOLOモデルの情報
        if hasattr(model, 'names'):
            info['class_names'] = model.names
        
        return info
    
    def _find_training_dir(self, model_path: Path) -> Optional[Path]:
        """モデルのトレーニングディレクトリを探す"""
        # モデルが weights/best.pt の形式の場合
        if model_path.parent.name == 'weights':
            return model_path.parent.parent
        
        # 同じディレクトリ内を探す
        parent_dir = model_path.parent
        for item in parent_dir.iterdir():
            if item.is_dir() and (item / 'results.csv').exists():
                return item
        
        return None
    
    def _visualize_confusion_matrix(self, training_dir: Path, output_dir: Path) -> Optional[Path]:
        """混同行列を可視化"""
        # 既存の混同行列をコピー
        cm_files = ['confusion_matrix.png', 'confusion_matrix_normalized.png']
        
        for cm_file in cm_files:
            src = training_dir / cm_file
            if src.exists():
                dst = output_dir / cm_file
                import shutil
                shutil.copy2(src, dst)
                return dst
        
        return None
    
    def _visualize_pr_curve(self, training_dir: Path, output_dir: Path) -> Optional[Path]:
        """PR曲線を可視化"""
        pr_file = training_dir / 'PR_curve.png'
        if pr_file.exists():
            dst = output_dir / 'PR_curve.png'
            import shutil
            shutil.copy2(pr_file, dst)
            return dst
        
        return None
    
    def _visualize_f1_curve(self, training_dir: Path, output_dir: Path) -> Optional[Path]:
        """F1曲線を可視化"""
        f1_file = training_dir / 'F1_curve.png'
        if f1_file.exists():
            dst = output_dir / 'F1_curve.png'
            import shutil
            shutil.copy2(f1_file, dst)
            return dst
        
        return None
    
    def _get_training_history(self, training_dir: Path, output_dir: Path) -> Optional[Dict[str, List[float]]]:
        """トレーニング履歴を取得して可視化"""
        results_file = training_dir / 'results.csv'
        if not results_file.exists():
            return None
        
        # CSVを読み込み
        df = pd.read_csv(results_file)
        
        # 主要なメトリクスを抽出
        history = {}
        metrics_map = {
            'epoch': 'epoch',
            'train/box_loss': 'train_box_loss',
            'train/cls_loss': 'train_cls_loss',
            'train/dfl_loss': 'train_dfl_loss',
            'val/box_loss': 'val_box_loss',
            'val/cls_loss': 'val_cls_loss',
            'val/dfl_loss': 'val_dfl_loss',
            'metrics/precision(B)': 'precision',
            'metrics/recall(B)': 'recall',
            'metrics/mAP50(B)': 'mAP50',
            'metrics/mAP50-95(B)': 'mAP50-95'
        }
        
        for csv_col, hist_key in metrics_map.items():
            if csv_col in df.columns:
                history[hist_key] = df[csv_col].fillna(0).tolist()
        
        # 学習曲線を可視化
        if history:
            self._plot_training_curves(history, output_dir)
        
        return history
    
    def _plot_training_curves(self, history: Dict[str, List[float]], output_dir: Path):
        """学習曲線をプロット"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training History', fontsize=16)
        
        # Loss curves
        ax = axes[0, 0]
        if 'epoch' in history:
            epochs = history['epoch']
            if 'train_box_loss' in history:
                ax.plot(epochs, history['train_box_loss'], label='Train Box Loss', alpha=0.8)
            if 'train_cls_loss' in history:
                ax.plot(epochs, history['train_cls_loss'], label='Train Class Loss', alpha=0.8)
            if 'val_box_loss' in history:
                ax.plot(epochs, history['val_box_loss'], label='Val Box Loss', linestyle='--', alpha=0.8)
            if 'val_cls_loss' in history:
                ax.plot(epochs, history['val_cls_loss'], label='Val Class Loss', linestyle='--', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # mAP curves
        ax = axes[0, 1]
        if 'epoch' in history:
            if 'mAP50' in history:
                ax.plot(epochs, history['mAP50'], label='mAP@0.5', color='green', linewidth=2)
            if 'mAP50-95' in history:
                ax.plot(epochs, history['mAP50-95'], label='mAP@0.5:0.95', color='blue', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.set_title('Mean Average Precision')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Precision/Recall curves
        ax = axes[1, 0]
        if 'epoch' in history:
            if 'precision' in history:
                ax.plot(epochs, history['precision'], label='Precision', color='orange', linewidth=2)
            if 'recall' in history:
                ax.plot(epochs, history['recall'], label='Recall', color='red', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Precision and Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = "Training Summary\n" + "="*30 + "\n"
        if 'mAP50' in history and history['mAP50']:
            summary_text += f"Best mAP@0.5: {max(history['mAP50']):.4f}\n"
            summary_text += f"Final mAP@0.5: {history['mAP50'][-1]:.4f}\n"
        if 'mAP50-95' in history and history['mAP50-95']:
            summary_text += f"Best mAP@0.5:0.95: {max(history['mAP50-95']):.4f}\n"
            summary_text += f"Final mAP@0.5:0.95: {history['mAP50-95'][-1]:.4f}\n"
        if 'precision' in history and history['precision']:
            summary_text += f"Final Precision: {history['precision'][-1]:.4f}\n"
        if 'recall' in history and history['recall']:
            summary_text += f"Final Recall: {history['recall'][-1]:.4f}\n"
        
        ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_class_metrics(self, training_dir: Path, model: YOLO) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """クラス別メトリクスを計算"""
        class_metrics = {}
        overall_metrics = {}
        
        # バリデーション結果から取得を試みる
        val_dir = training_dir / 'val'
        if not val_dir.exists():
            # 代わりにモデルで検証を実行することもできる
            return class_metrics, overall_metrics
        
        # クラス名を取得
        class_names = []
        if hasattr(model, 'names'):
            class_names = list(model.names.values()) if isinstance(model.names, dict) else model.names
        
        # TODO: 実際のメトリクス計算
        # ここでは仮の値を返す
        for i, class_name in enumerate(class_names):
            class_metrics[class_name] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'ap50': 0.0,
                'ap': 0.0
            }
        
        overall_metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'map50': 0.0,
            'map50_95': 0.0
        }
        
        return class_metrics, overall_metrics
    
    def _visualize_class_metrics(self, class_metrics: Dict[str, Dict[str, float]], output_dir: Path):
        """クラス別メトリクスを可視化"""
        if not class_metrics:
            return
        
        # データフレームに変換
        df = pd.DataFrame(class_metrics).T
        
        # バープロット
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Class-wise Performance Metrics', fontsize=16)
        
        metrics = ['precision', 'recall', 'f1_score', 'ap50']
        titles = ['Precision', 'Recall', 'F1-Score', 'AP@0.5']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i // 2, i % 2]
            if metric in df.columns:
                df[metric].plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
                ax.set_title(title, fontsize=14)
                ax.set_xlabel('Class', fontsize=12)
                ax.set_ylabel(title, fontsize=12)
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                
                # 値をバーの上に表示
                for j, v in enumerate(df[metric]):
                    ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'class_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()


# シングルトンインスタンス
_model_visualizer: Optional[ModelVisualizer] = None


def get_model_visualizer() -> ModelVisualizer:
    """モデル可視化器のインスタンスを取得"""
    global _model_visualizer
    if _model_visualizer is None:
        _model_visualizer = ModelVisualizer()
    return _model_visualizer