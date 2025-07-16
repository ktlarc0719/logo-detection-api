"""
データセット検証モジュール
"""

import os
import yaml
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import cv2
import numpy as np

from src.models.ml_system_schemas import (
    DatasetValidationRequest, DatasetValidationResult, DatasetType
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetValidator:
    """データセット検証クラス"""
    
    def __init__(self):
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        self.label_format = '.txt'
    
    def validate_dataset(self, request: DatasetValidationRequest) -> DatasetValidationResult:
        """データセットを検証"""
        logger.info(f"Validating dataset: {request.dataset_path}")
        
        dataset_path = Path(request.dataset_path)
        if not dataset_path.exists():
            return DatasetValidationResult(
                dataset_path=str(dataset_path),
                valid=False,
                total_images=0,
                total_labels=0,
                classes=[],
                class_distribution={},
                errors=[f"Dataset path does not exist: {dataset_path}"]
            )
        
        # dataset.yamlを読み込み
        yaml_path = dataset_path / "dataset.yaml"
        if not yaml_path.exists():
            yaml_path = dataset_path / "data.yaml"
        
        if not yaml_path.exists():
            return DatasetValidationResult(
                dataset_path=str(dataset_path),
                valid=False,
                total_images=0,
                total_labels=0,
                classes=[],
                class_distribution={},
                errors=["dataset.yaml or data.yaml not found"]
            )
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                dataset_config = yaml.safe_load(f)
        except Exception as e:
            return DatasetValidationResult(
                dataset_path=str(dataset_path),
                valid=False,
                total_images=0,
                total_labels=0,
                classes=[],
                class_distribution={},
                errors=[f"Failed to load YAML: {str(e)}"]
            )
        
        # クラス情報を取得
        classes = self._get_classes(dataset_config)
        
        # 各分割を検証
        errors = []
        warnings = []
        class_distribution = {}
        split_info = {}
        total_images = 0
        total_labels = 0
        
        for split in [DatasetType.TRAIN, DatasetType.VAL, DatasetType.TEST]:
            split_path = dataset_config.get(split.value)
            if not split_path:
                warnings.append(f"No {split.value} split defined")
                continue
            
            # 相対パスを絶対パスに変換
            if not os.path.isabs(split_path):
                split_path = dataset_path / split_path
            else:
                split_path = Path(split_path)
            
            # 分割を検証
            info = self._validate_split(
                split_path, 
                classes, 
                split.value,
                request.check_images,
                request.check_labels,
                request.validate_format
            )
            
            split_info[f"{split.value}_info"] = info
            class_distribution[split.value] = info['class_distribution']
            total_images += info['num_images']
            total_labels += info['num_labels']
            
            errors.extend(info.get('errors', []))
            warnings.extend(info.get('warnings', []))
        
        # 統計情報を計算
        statistics = self._calculate_statistics(
            class_distribution, classes, split_info
        )
        
        # 有効性を判定
        valid = len(errors) == 0 and total_images > 0 and total_labels > 0
        
        return DatasetValidationResult(
            dataset_path=str(dataset_path),
            valid=valid,
            total_images=total_images,
            total_labels=total_labels,
            train_info=split_info.get('train_info'),
            val_info=split_info.get('val_info'),
            test_info=split_info.get('test_info'),
            classes=classes,
            class_distribution=class_distribution,
            errors=errors,
            warnings=warnings,
            statistics=statistics
        )
    
    def _get_classes(self, dataset_config: Dict) -> List[str]:
        """データセット設定からクラス名を取得"""
        names = dataset_config.get('names', [])
        if isinstance(names, list):
            return names
        elif isinstance(names, dict):
            return [names[i] for i in sorted(names.keys())]
        return []
    
    def _validate_split(
        self,
        split_path: Path,
        classes: List[str],
        split_name: str,
        check_images: bool,
        check_labels: bool,
        validate_format: bool
    ) -> Dict[str, Any]:
        """データセットの分割を検証"""
        info = {
            'path': str(split_path),
            'exists': split_path.exists(),
            'num_images': 0,
            'num_labels': 0,
            'class_distribution': defaultdict(int),
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        if not split_path.exists():
            info['errors'].append(f"{split_name} directory does not exist: {split_path}")
            return info
        
        # 画像ディレクトリとラベルディレクトリを確認
        images_dir = split_path if 'images' in str(split_path) else split_path / 'images'
        labels_dir = str(images_dir).replace('images', 'labels')
        labels_dir = Path(labels_dir)
        
        if not images_dir.exists():
            info['errors'].append(f"Images directory not found: {images_dir}")
            return info
        
        if check_labels and not labels_dir.exists():
            info['errors'].append(f"Labels directory not found: {labels_dir}")
            return info
        
        # 画像ファイルを収集
        image_files = []
        for ext in self.supported_image_formats:
            image_files.extend(glob.glob(str(images_dir / f"*{ext}")))
            image_files.extend(glob.glob(str(images_dir / f"*{ext.upper()}")))
        
        info['num_images'] = len(image_files)
        
        if check_images and info['num_images'] == 0:
            info['warnings'].append(f"No images found in {images_dir}")
        
        # ラベルファイルを検証
        if check_labels:
            label_files = glob.glob(str(labels_dir / "*.txt"))
            info['num_labels'] = len(label_files)
            
            # 画像とラベルの対応を確認
            images_without_labels = 0
            labels_without_images = 0
            valid_pairs = 0
            
            image_basenames = {Path(f).stem for f in image_files}
            label_basenames = {Path(f).stem for f in label_files}
            
            images_without_labels = len(image_basenames - label_basenames)
            labels_without_images = len(label_basenames - image_basenames)
            valid_pairs = len(image_basenames & label_basenames)
            
            if images_without_labels > 0:
                info['warnings'].append(
                    f"{images_without_labels} images without corresponding labels"
                )
            
            if labels_without_images > 0:
                info['warnings'].append(
                    f"{labels_without_images} labels without corresponding images"
                )
            
            info['statistics']['valid_pairs'] = valid_pairs
            info['statistics']['images_without_labels'] = images_without_labels
            info['statistics']['labels_without_images'] = labels_without_images
            
            # クラス分布を計算
            if validate_format:
                empty_labels = 0
                invalid_labels = 0
                
                for label_file in label_files[:min(len(label_files), 1000)]:  # 最大1000ファイルまで
                    try:
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                            
                        if not lines:
                            empty_labels += 1
                            continue
                        
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:  # YOLO format: class x y w h
                                try:
                                    class_id = int(parts[0])
                                    if 0 <= class_id < len(classes):
                                        class_name = classes[class_id]
                                        info['class_distribution'][class_name] += 1
                                    else:
                                        invalid_labels += 1
                                except ValueError:
                                    invalid_labels += 1
                            else:
                                invalid_labels += 1
                    
                    except Exception as e:
                        info['errors'].append(f"Error reading label file {label_file}: {str(e)}")
                
                if empty_labels > 0:
                    info['warnings'].append(f"{empty_labels} empty label files")
                
                if invalid_labels > 0:
                    info['errors'].append(f"{invalid_labels} invalid label entries")
                
                info['statistics']['empty_labels'] = empty_labels
                info['statistics']['invalid_labels'] = invalid_labels
        
        # defaultdictを通常のdictに変換
        info['class_distribution'] = dict(info['class_distribution'])
        
        # 画像サイズの統計（サンプリング）
        if check_images and validate_format and image_files:
            sample_size = min(100, len(image_files))
            sample_files = np.random.choice(image_files, sample_size, replace=False)
            
            widths = []
            heights = []
            corrupted = 0
            
            for img_file in sample_files:
                try:
                    img = cv2.imread(img_file)
                    if img is not None:
                        h, w = img.shape[:2]
                        heights.append(h)
                        widths.append(w)
                    else:
                        corrupted += 1
                except Exception:
                    corrupted += 1
            
            if widths:
                info['statistics']['avg_width'] = float(np.mean(widths))
                info['statistics']['avg_height'] = float(np.mean(heights))
                info['statistics']['min_width'] = int(np.min(widths))
                info['statistics']['max_width'] = int(np.max(widths))
                info['statistics']['min_height'] = int(np.min(heights))
                info['statistics']['max_height'] = int(np.max(heights))
            
            if corrupted > 0:
                info['errors'].append(f"{corrupted} corrupted images in sample")
        
        return info
    
    def _calculate_statistics(
        self,
        class_distribution: Dict[str, Dict[str, int]],
        classes: List[str],
        split_info: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """データセット全体の統計を計算"""
        statistics = {
            'total_instances': 0,
            'class_balance': {},
            'split_balance': {},
            'recommendations': []
        }
        
        # クラスごとのインスタンス数を集計
        class_totals = defaultdict(int)
        for split, dist in class_distribution.items():
            for class_name, count in dist.items():
                class_totals[class_name] += count
                statistics['total_instances'] += count
        
        # total_instancesをint型に変換
        statistics['total_instances'] = int(statistics['total_instances'])
        
        # クラスバランスを計算
        if statistics['total_instances'] > 0:
            for class_name in classes:
                count = int(class_totals[class_name])  # int型に変換
                percentage = float((count / statistics['total_instances']) * 100)  # float型に変換
                statistics['class_balance'][class_name] = {
                    'count': count,
                    'percentage': percentage
                }
                
                # 推奨事項
                if percentage < 10:
                    statistics['recommendations'].append(
                        f"{class_name} has only {percentage:.1f}% of data. Consider adding more samples."
                    )
                elif percentage > 50:
                    statistics['recommendations'].append(
                        f"{class_name} dominates with {percentage:.1f}% of data. Consider balancing."
                    )
        
        # 分割バランスを計算
        split_totals = {}
        for split in ['train', 'val', 'test']:
            if split in class_distribution:
                total = sum(class_distribution[split].values())
                split_totals[split] = total
        
        total_split = sum(split_totals.values())
        if total_split > 0:
            for split, count in split_totals.items():
                percentage = float((count / total_split) * 100)  # float型に変換
                statistics['split_balance'][split] = {
                    'count': int(count),  # int型に変換
                    'percentage': percentage
                }
            
            # 推奨される分割比率のチェック
            train_pct = statistics['split_balance'].get('train', {}).get('percentage', 0)
            val_pct = statistics['split_balance'].get('val', {}).get('percentage', 0)
            test_pct = statistics['split_balance'].get('test', {}).get('percentage', 0)
            
            if train_pct < 60:
                statistics['recommendations'].append(
                    "Training set is less than 60% of data. Consider increasing."
                )
            if val_pct < 10:
                statistics['recommendations'].append(
                    "Validation set is less than 10% of data. May affect model selection."
                )
            if test_pct < 10:
                statistics['recommendations'].append(
                    "Test set is less than 10% of data. May not be representative."
                )
        
        return statistics


# シングルトンインスタンス
_dataset_validator: Optional[DatasetValidator] = None


def get_dataset_validator() -> DatasetValidator:
    """データセット検証器のインスタンスを取得"""
    global _dataset_validator
    if _dataset_validator is None:
        _dataset_validator = DatasetValidator()
    return _dataset_validator