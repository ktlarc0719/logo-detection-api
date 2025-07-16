"""
モデル検証モジュール
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import cv2
from collections import defaultdict

from ultralytics import YOLO
import torch

from src.models.ml_system_schemas import (
    ModelValidationRequest, ModelValidationResult, PredictionResult
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelValidator:
    """モデル検証クラス"""
    
    def __init__(self):
        self.output_dir = Path("ml_outputs/validations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = '0' if torch.cuda.is_available() else 'cpu'
    
    def validate_model(self, request: ModelValidationRequest) -> ModelValidationResult:
        """モデルを検証"""
        logger.info(f"Validating model: {request.model_path}")
        
        # モデルをロード
        model_path = Path(request.model_path)
        if not model_path.exists():
            raise ValueError(f"Model not found: {model_path}")
        
        try:
            model = YOLO(str(model_path))
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
        
        # テストデータセットを確認
        test_dataset_path = Path(request.test_dataset_path)
        if not test_dataset_path.exists():
            raise ValueError(f"Test dataset not found: {test_dataset_path}")
        
        # 出力ディレクトリを作成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        val_dir = self.output_dir / f"{model_path.stem}_{timestamp}"
        val_dir.mkdir(exist_ok=True)
        
        # 画像とラベルのパスを取得
        images_dir = test_dataset_path / "images"
        labels_dir = test_dataset_path / "labels"
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        
        # クラス名を取得
        class_names = self._get_class_names(model)
        
        # 検証を実行
        start_time = datetime.now()
        prediction_results = []
        
        # 画像ファイルを取得
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        total_images = len(image_files)
        
        if total_images == 0:
            raise ValueError("No images found in test dataset")
        
        # 予測を実行
        correct_predictions = 0
        class_metrics_raw = defaultdict(lambda: {
            'tp': 0, 'fp': 0, 'fn': 0, 'total_iou': 0.0, 'count': 0
        })
        
        for idx, img_path in enumerate(image_files):
            # 進捗ログ
            if idx % 10 == 0:
                logger.info(f"Processing image {idx+1}/{total_images}")
            
            # 正解ラベルを読み込み
            label_path = labels_dir / f"{img_path.stem}.txt"
            ground_truth = self._load_labels(label_path, class_names)
            
            # 予測を実行
            results = model.predict(
                str(img_path),
                conf=request.confidence_threshold,
                iou=request.iou_threshold,
                device=self.device,
                verbose=False
            )
            
            # 結果を処理
            predictions = self._process_predictions(results[0], class_names)
            
            # 評価
            is_correct, error_type, error_details, class_results = self._evaluate_prediction(
                ground_truth, predictions, class_names, request.iou_threshold
            )
            
            if is_correct:
                correct_predictions += 1
            
            # クラス別メトリクスを更新
            for class_name, metrics in class_results.items():
                class_metrics_raw[class_name]['tp'] += metrics['tp']
                class_metrics_raw[class_name]['fp'] += metrics['fp']
                class_metrics_raw[class_name]['fn'] += metrics['fn']
                class_metrics_raw[class_name]['total_iou'] += metrics['iou_sum']
                class_metrics_raw[class_name]['count'] += metrics['tp']
            
            # 予測結果を保存
            pred_result = PredictionResult(
                image_path=str(img_path),
                ground_truth=ground_truth,
                predictions=predictions,
                correct=is_correct,
                confidence_scores=[p['confidence'] for p in predictions],
                iou_scores=[],
                error_type=error_type,
                error_details=error_details
            )
            prediction_results.append(pred_result)
            
            # エラーサンプルを保存
            if request.save_predictions and not is_correct and request.analyze_errors:
                self._save_error_sample(img_path, pred_result, val_dir)
        
        # メトリクスを計算
        accuracy = correct_predictions / total_images if total_images > 0 else 0.0
        
        # クラス別メトリクスを計算
        class_metrics = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for class_name, raw_metrics in class_metrics_raw.items():
            tp = raw_metrics['tp']
            fp = raw_metrics['fp']
            fn = raw_metrics['fn']
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            avg_iou = raw_metrics['total_iou'] / raw_metrics['count'] if raw_metrics['count'] > 0 else 0.0
            
            class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'average_iou': avg_iou
            }
        
        # 全体メトリクスを計算
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) \
                     if (overall_precision + overall_recall) > 0 else 0.0
        
        # エラー分析
        error_analysis = self._analyze_errors(prediction_results) if request.analyze_errors else {}
        
        # 混同行列を生成
        confusion_matrix_path = None
        if request.save_confusion_matrix:
            confusion_matrix_path = self._generate_confusion_matrix(
                prediction_results, class_names, val_dir
            )
        
        # 処理時間を計算
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ModelValidationResult(
            model_path=str(model_path),
            test_dataset_path=str(test_dataset_path),
            accuracy=accuracy,
            precision=overall_precision,
            recall=overall_recall,
            f1_score=overall_f1,
            map50=0.0,  # TODO: 実際のmAP計算を実装
            map50_95=0.0,  # TODO: 実際のmAP計算を実装
            class_metrics=class_metrics,
            total_images=total_images,
            correct_predictions=correct_predictions,
            prediction_results=prediction_results if request.save_predictions else [],
            error_analysis=error_analysis,
            common_errors=self._get_common_errors(error_analysis),
            confusion_matrix_path=str(confusion_matrix_path) if confusion_matrix_path else None,
            error_samples_dir=str(val_dir / "error_samples") if request.analyze_errors else None,
            confidence_threshold=request.confidence_threshold,
            iou_threshold=request.iou_threshold,
            processing_time=processing_time
        )
    
    def _get_class_names(self, model: YOLO) -> List[str]:
        """モデルからクラス名を取得"""
        if hasattr(model, 'names'):
            if isinstance(model.names, dict):
                return [model.names[i] for i in sorted(model.names.keys())]
            else:
                return list(model.names)
        return []
    
    def _load_labels(self, label_path: Path, class_names: List[str]) -> List[Dict[str, Any]]:
        """ラベルファイルを読み込み"""
        labels = []
        
        if not label_path.exists():
            return labels
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                if 0 <= class_id < len(class_names):
                    labels.append({
                        'class': class_names[class_id],
                        'class_id': class_id,
                        'x_center': float(parts[1]),
                        'y_center': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4])
                    })
        
        return labels
    
    def _process_predictions(self, result, class_names: List[str]) -> List[Dict[str, Any]]:
        """予測結果を処理"""
        predictions = []
        
        if result.boxes is None:
            return predictions
        
        boxes = result.boxes
        for i in range(len(boxes)):
            box = boxes[i]
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            # 画像サイズで正規化
            img_h, img_w = result.orig_shape
            x_center = ((xyxy[0] + xyxy[2]) / 2) / img_w
            y_center = ((xyxy[1] + xyxy[3]) / 2) / img_h
            width = (xyxy[2] - xyxy[0]) / img_w
            height = (xyxy[3] - xyxy[1]) / img_h
            
            predictions.append({
                'class': class_names[cls] if cls < len(class_names) else f'class_{cls}',
                'class_id': cls,
                'confidence': float(conf),
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
        
        return predictions
    
    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """IoUを計算"""
        # Convert center format to corner format
        x1_min = box1['x_center'] - box1['width'] / 2
        y1_min = box1['y_center'] - box1['height'] / 2
        x1_max = box1['x_center'] + box1['width'] / 2
        y1_max = box1['y_center'] + box1['height'] / 2
        
        x2_min = box2['x_center'] - box2['width'] / 2
        y2_min = box2['y_center'] - box2['height'] / 2
        x2_max = box2['x_center'] + box2['width'] / 2
        y2_max = box2['y_center'] + box2['height'] / 2
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Calculate union
        area1 = box1['width'] * box1['height']
        area2 = box2['width'] * box2['height']
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _evaluate_prediction(
        self,
        ground_truth: List[Dict],
        predictions: List[Dict],
        class_names: List[str],
        iou_threshold: float
    ) -> Tuple[bool, Optional[str], Optional[str], Dict[str, Dict[str, int]]]:
        """予測を評価"""
        class_results = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'iou_sum': 0.0})
        
        # マッチング用のフラグ
        gt_matched = [False] * len(ground_truth)
        pred_matched = [False] * len(predictions)
        
        # 予測と正解のマッチング
        for pred_idx, pred in enumerate(predictions):
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_matched[gt_idx]:
                    continue
                
                if pred['class'] == gt['class']:
                    iou = self._calculate_iou(pred, gt)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                # True Positive
                gt_matched[best_gt_idx] = True
                pred_matched[pred_idx] = True
                class_results[pred['class']]['tp'] += 1
                class_results[pred['class']]['iou_sum'] += best_iou
            else:
                # False Positive
                class_results[pred['class']]['fp'] += 1
        
        # False Negatives
        for gt_idx, gt in enumerate(ground_truth):
            if not gt_matched[gt_idx]:
                class_results[gt['class']]['fn'] += 1
        
        # 全体の正誤判定
        is_correct = all(gt_matched) and all(pred_matched) and len(ground_truth) == len(predictions)
        
        # エラータイプの判定
        error_type = None
        error_details = None
        
        if not is_correct:
            if len(predictions) == 0 and len(ground_truth) > 0:
                error_type = "false_negative"
                error_details = f"Missed {len(ground_truth)} objects"
            elif len(predictions) > 0 and len(ground_truth) == 0:
                error_type = "false_positive"
                error_details = f"Detected {len(predictions)} objects when none exist"
            elif any(not matched for matched in pred_matched):
                error_type = "false_positive"
                error_details = f"{sum(1 for m in pred_matched if not m)} false detections"
            elif any(not matched for matched in gt_matched):
                error_type = "false_negative"
                error_details = f"{sum(1 for m in gt_matched if not m)} missed detections"
            else:
                error_type = "misclassification"
                error_details = "Class mismatch"
        
        return is_correct, error_type, error_details, dict(class_results)
    
    def _save_error_sample(self, img_path: Path, result: PredictionResult, output_dir: Path):
        """エラーサンプルを保存"""
        error_dir = output_dir / "error_samples" / result.error_type
        error_dir.mkdir(parents=True, exist_ok=True)
        
        # 画像をコピー
        dst_path = error_dir / img_path.name
        shutil.copy2(img_path, dst_path)
        
        # 結果をJSONで保存
        json_path = error_dir / f"{img_path.stem}_result.json"
        with open(json_path, 'w') as f:
            json.dump({
                'image_path': str(img_path),
                'ground_truth': result.ground_truth,
                'predictions': result.predictions,
                'error_type': result.error_type,
                'error_details': result.error_details
            }, f, indent=2)
    
    def _analyze_errors(self, prediction_results: List[PredictionResult]) -> Dict[str, Any]:
        """エラーを分析"""
        error_counts = defaultdict(int)
        error_by_class = defaultdict(lambda: defaultdict(int))
        
        for result in prediction_results:
            if not result.correct and result.error_type:
                error_counts[result.error_type] += 1
                
                # クラス別エラーを集計
                for gt in result.ground_truth:
                    error_by_class[gt['class']][result.error_type] += 1
        
        return {
            'error_counts': dict(error_counts),
            'error_by_class': {k: dict(v) for k, v in error_by_class.items()},
            'total_errors': sum(error_counts.values())
        }
    
    def _get_common_errors(self, error_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """よくあるエラーを取得"""
        if not error_analysis:
            return []
        
        common_errors = []
        error_by_class = error_analysis.get('error_by_class', {})
        
        for class_name, errors in error_by_class.items():
            for error_type, count in errors.items():
                common_errors.append({
                    'class': class_name,
                    'error_type': error_type,
                    'count': count
                })
        
        # カウントで降順ソート
        common_errors.sort(key=lambda x: x['count'], reverse=True)
        return common_errors[:10]  # トップ10のエラー
    
    def _generate_confusion_matrix(
        self,
        prediction_results: List[PredictionResult],
        class_names: List[str],
        output_dir: Path
    ) -> Path:
        """混同行列を生成"""
        import matplotlib.pyplot as plt
        
        n_classes = len(class_names)
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        for result in prediction_results:
            # 各画像の予測と正解のペアをカウント
            gt_classes = [gt['class_id'] for gt in result.ground_truth]
            pred_classes = [pred['class_id'] for pred in result.predictions]
            
            # 簡易的な実装：最初の検出のみを使用
            if gt_classes and pred_classes:
                confusion_matrix[gt_classes[0], pred_classes[0]] += 1
            elif gt_classes and not pred_classes:
                # False negative - 背景として分類
                pass
            elif not gt_classes and pred_classes:
                # False positive
                pass
        
        # プロット
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # ヒートマップを手動で作成
        im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')
        
        # カラーバーを追加
        plt.colorbar(im, ax=ax)
        
        # ラベルを設定
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        
        # ラベルを回転
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 各セルに値を表示
        for i in range(n_classes):
            for j in range(n_classes):
                text = ax.text(j, i, confusion_matrix[i, j],
                              ha="center", va="center", color="black" if confusion_matrix[i, j] < confusion_matrix.max() / 2 else "white")
        
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.tight_layout()
        
        output_path = output_dir / 'confusion_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path


# シングルトンインスタンス
_model_validator: Optional[ModelValidator] = None


def get_model_validator() -> ModelValidator:
    """モデル検証器のインスタンスを取得"""
    global _model_validator
    if _model_validator is None:
        _model_validator = ModelValidator()
    return _model_validator