import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dataset", tags=["dataset"])


class DatasetSplitRequest(BaseModel):
    source_images_path: str  # アノテーション済み画像のディレクトリ
    source_labels_path: str  # labelImgで作成したラベルのディレクトリ
    target_dataset_path: str  # 出力先のデータセットパス
    train_ratio: float = 0.8  # トレーニングデータの割合（デフォルト8割）
    seed: int = 42  # ランダムシードの固定


class DatasetSplitResponse(BaseModel):
    success: bool
    message: str
    train_count: int
    val_count: int
    dataset_path: str


def create_dataset_structure(base_path: str) -> Dict[str, Path]:
    """YOLOデータセット構造を作成"""
    paths = {
        'base': Path(base_path),
        'images_train': Path(base_path) / 'train' / 'images',
        'images_val': Path(base_path) / 'val' / 'images',
        'labels_train': Path(base_path) / 'train' / 'labels',
        'labels_val': Path(base_path) / 'val' / 'labels',
    }
    
    # ディレクトリを作成
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths


def get_image_label_pairs(images_dir: str, labels_dir: str) -> List[Tuple[Path, Path]]:
    """画像とラベルのペアを取得"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    pairs = []
    
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    for image_file in images_path.iterdir():
        if image_file.suffix.lower() in image_extensions:
            # 対応するラベルファイルを探す
            label_file = labels_path / f"{image_file.stem}.txt"
            if label_file.exists():
                pairs.append((image_file, label_file))
            else:
                logger.warning(f"Label file not found for {image_file.name}")
    
    return pairs


def split_dataset(pairs: List[Tuple[Path, Path]], train_ratio: float, seed: int) -> Tuple[List, List]:
    """データセットをトレーニングと検証に分割"""
    random.seed(seed)
    shuffled_pairs = pairs.copy()
    random.shuffle(shuffled_pairs)
    
    split_idx = int(len(shuffled_pairs) * train_ratio)
    train_pairs = shuffled_pairs[:split_idx]
    val_pairs = shuffled_pairs[split_idx:]
    
    return train_pairs, val_pairs


def copy_files(pairs: List[Tuple[Path, Path]], target_images: Path, target_labels: Path) -> None:
    """ファイルをコピー"""
    for image_file, label_file in pairs:
        shutil.copy2(image_file, target_images / image_file.name)
        shutil.copy2(label_file, target_labels / label_file.name)


def create_dataset_yaml(dataset_path: Path, classes_file: Path = None) -> None:
    """dataset.yamlを作成"""
    # classes.txtがあれば読み込む
    if classes_file and classes_file.exists():
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f if line.strip()]
    else:
        # デフォルトのクラス名
        classes = ['logo']
    
    yaml_content = f"""# YOLOv5/v8 Dataset Configuration
path: {dataset_path.absolute()}
train: train/images
val: val/images

# Classes
nc: {len(classes)}
names: {classes}
"""
    
    yaml_path = dataset_path / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)


@router.post("/split", response_model=DatasetSplitResponse)
async def split_dataset_endpoint(request: DatasetSplitRequest):
    """画像とラベルを自動的にtrain/valに分割"""
    try:
        # パスの検証
        if not os.path.exists(request.source_images_path):
            raise HTTPException(status_code=400, detail=f"Source images path not found: {request.source_images_path}")
        
        if not os.path.exists(request.source_labels_path):
            raise HTTPException(status_code=400, detail=f"Source labels path not found: {request.source_labels_path}")
        
        # データセット構造を作成
        paths = create_dataset_structure(request.target_dataset_path)
        
        # 画像とラベルのペアを取得
        pairs = get_image_label_pairs(request.source_images_path, request.source_labels_path)
        
        if not pairs:
            raise HTTPException(status_code=400, detail="No image-label pairs found")
        
        # データセットを分割
        train_pairs, val_pairs = split_dataset(pairs, request.train_ratio, request.seed)
        
        # ファイルをコピー
        copy_files(train_pairs, paths['images_train'], paths['labels_train'])
        copy_files(val_pairs, paths['images_val'], paths['labels_val'])
        
        # classes.txtもコピー（存在する場合）
        classes_file = Path(request.source_labels_path) / 'classes.txt'
        if classes_file.exists():
            shutil.copy2(classes_file, paths['base'] / 'classes.txt')
        
        # dataset.yamlを作成
        create_dataset_yaml(paths['base'], classes_file)
        
        logger.info(f"Dataset split completed: {len(train_pairs)} train, {len(val_pairs)} val")
        
        return DatasetSplitResponse(
            success=True,
            message=f"Dataset successfully split into {len(train_pairs)} training and {len(val_pairs)} validation images",
            train_count=len(train_pairs),
            val_count=len(val_pairs),
            dataset_path=str(paths['base'])
        )
        
    except Exception as e:
        logger.error(f"Error splitting dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate/{dataset_path:path}")
async def validate_dataset(dataset_path: str):
    """データセットの構造を検証"""
    try:
        base_path = Path(dataset_path)
        
        if not base_path.exists():
            raise HTTPException(status_code=404, detail="Dataset path not found")
        
        # 必要なディレクトリの存在確認
        required_dirs = [
            base_path / 'train' / 'images',
            base_path / 'val' / 'images',
            base_path / 'train' / 'labels',
            base_path / 'val' / 'labels',
        ]
        
        missing_dirs = [str(d) for d in required_dirs if not d.exists()]
        
        # ファイル数をカウント
        train_images = len(list((base_path / 'train' / 'images').glob('*')))
        train_labels = len(list((base_path / 'train' / 'labels').glob('*.txt')))
        val_images = len(list((base_path / 'val' / 'images').glob('*')))
        val_labels = len(list((base_path / 'val' / 'labels').glob('*.txt')))
        
        return {
            "valid": len(missing_dirs) == 0,
            "missing_directories": missing_dirs,
            "statistics": {
                "train_images": train_images,
                "train_labels": train_labels,
                "val_images": val_images,
                "val_labels": val_labels,
            },
            "dataset_yaml_exists": (base_path / 'dataset.yaml').exists(),
            "classes_txt_exists": (base_path / 'classes.txt').exists(),
        }
        
    except Exception as e:
        logger.error(f"Error validating dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))