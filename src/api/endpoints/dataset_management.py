"""
データセット管理APIエンドポイント
"""

import os
import shutil
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/dataset", tags=["Dataset Management"])


class DatasetSplitRequest(BaseModel):
    """データセットスプリットリクエスト"""
    class_name: str = Field(..., description="クラス名（例: BANDAI）")
    base_path: str = Field("/mnt/c/02_yolo_projects", description="ベースパス")
    destination_dataset: str = Field(..., description="コピー先データセット名")
    create_new: bool = Field(False, description="新規データセット作成フラグ")
    train_ratio: float = Field(0.69, ge=0, le=1, description="Train比率")
    val_ratio: float = Field(0.17, ge=0, le=1, description="Validation比率")
    test_ratio: float = Field(0.14, ge=0, le=1, description="Test比率")
    random_seed: int = Field(42, description="ランダムシード")
    max_images: Optional[int] = Field(None, ge=1, description="処理する最大画像数（未指定時は全件）")


class DatasetSplitResponse(BaseModel):
    """データセットスプリットレスポンス"""
    success: bool
    train_count: int
    val_count: int
    test_count: int
    skipped_count: int
    total_processed: int
    train_percentage: float
    val_percentage: float
    test_percentage: float


class DeleteClassRequest(BaseModel):
    """クラス削除リクエスト"""
    dataset_name: str = Field(..., description="データセット名")
    class_name: str = Field(..., description="削除するクラス名")


class DeleteClassResponse(BaseModel):
    """クラス削除レスポンス"""
    success: bool
    train_deleted: int
    val_deleted: int
    test_deleted: int
    total_deleted: int


class DeleteDatasetRequest(BaseModel):
    """データセット削除リクエスト"""
    dataset_name: str = Field(..., description="削除するデータセット名")
    confirm: bool = Field(..., description="削除確認フラグ")


class DeleteDatasetResponse(BaseModel):
    """データセット削除レスポンス"""
    success: bool
    deleted_path: str
    message: str


class DatasetInfo(BaseModel):
    """データセット情報"""
    name: str
    path: str
    exists: bool


class CreateDatasetRequest(BaseModel):
    """データセット作成リクエスト"""
    dataset_name: str = Field(..., description="作成するデータセット名")


class CreateDatasetResponse(BaseModel):
    """データセット作成レスポンス"""
    success: bool
    dataset_path: str
    message: str


# データセットのベースパス（設定可能にする）
DATASETS_BASE_PATH = Path("/root/projects/logo-detection-api/datasets")

# Create datasets directory if it doesn't exist
DATASETS_BASE_PATH.mkdir(parents=True, exist_ok=True)


def get_dataset_path(dataset_name: str) -> Path:
    """データセットパスを取得"""
    return DATASETS_BASE_PATH / dataset_name


def create_dataset_structure(dataset_path: Path, create_yaml: bool = True) -> Dict[str, Path]:
    """データセット構造を作成"""
    paths = {
        'base': dataset_path,
        'train_images': dataset_path / 'train' / 'images',
        'train_labels': dataset_path / 'train' / 'labels',
        'val_images': dataset_path / 'val' / 'images',
        'val_labels': dataset_path / 'val' / 'labels',
        'test_images': dataset_path / 'test' / 'images',
        'test_labels': dataset_path / 'test' / 'labels',
    }
    
    # ディレクトリを作成
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    # 新規作成時は空のdataset.yamlを作成
    if create_yaml:
        yaml_path = dataset_path / 'dataset.yaml'
        if not yaml_path.exists():
            yaml_content = f"""# YOLOv8 Dataset Configuration
path: {dataset_path.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
nc: 0
names: []
"""
            with open(yaml_path, 'w', encoding='utf-8') as f:
                f.write(yaml_content)
            logger.info(f"Created empty dataset.yaml at {yaml_path}")
    
    return paths


def get_labeled_images(images_dir: Path, labels_dir: Path, class_name: str) -> List[tuple]:
    """ラベル付き画像のリストを取得"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    labeled_images = []
    
    if not images_dir.exists():
        return labeled_images
        
    for image_file in images_dir.iterdir():
        if image_file.suffix.lower() in image_extensions:
            # 対応するラベルファイルを確認
            label_file = labels_dir / f"{image_file.stem}.txt"
            if label_file.exists():
                labeled_images.append((image_file, label_file, class_name))
    
    return labeled_images


def split_data(items: List[tuple], train_ratio: float, val_ratio: float, 
               test_ratio: float, seed: int) -> tuple:
    """データを指定比率で分割"""
    np.random.seed(seed)
    random.seed(seed)
    
    # シャッフル
    shuffled_items = items.copy()
    random.shuffle(shuffled_items)
    
    total = len(shuffled_items)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_items = shuffled_items[:train_size]
    val_items = shuffled_items[train_size:train_size + val_size]
    test_items = shuffled_items[train_size + val_size:]
    
    return train_items, val_items, test_items


def copy_with_rename(items: List[tuple], target_images: Path, target_labels: Path,
                     existing_files: set) -> tuple:
    """ファイルをコピーして名前を変更"""
    copied_count = 0
    skipped_count = 0
    
    for image_file, label_file, class_name in items:
        # 新しいファイル名を生成
        # ファイル名に既にクラス名が含まれているかチェック
        if image_file.stem.startswith(f"{class_name}_"):
            # 既にクラス名が含まれている場合はそのまま使用
            base_name = image_file.stem
        else:
            # クラス名が含まれていない場合のみ追加
            base_name = f"{class_name}_{image_file.stem}"
        new_image_name = f"{base_name}{image_file.suffix}"
        new_label_name = f"{base_name}.txt"
        
        # 既存ファイルチェック
        if new_image_name in existing_files:
            skipped_count += 1
            continue
        
        # コピー実行
        shutil.copy2(image_file, target_images / new_image_name)
        shutil.copy2(label_file, target_labels / new_label_name)
        existing_files.add(new_image_name)
        copied_count += 1
    
    return copied_count, skipped_count


@router.post("/create", response_model=CreateDatasetResponse)
async def create_dataset(request: CreateDatasetRequest):
    """新規データセットを作成（フォルダ構造とdataset.yaml）"""
    try:
        dataset_path = get_dataset_path(request.dataset_name)
        
        if dataset_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"データセット '{request.dataset_name}' は既に存在します"
            )
        
        # データセット構造を作成（dataset.yamlも作成）
        paths = create_dataset_structure(dataset_path, create_yaml=True)
        logger.info(f"Created new dataset: {request.dataset_name} at {dataset_path}")
        
        return CreateDatasetResponse(
            success=True,
            dataset_path=str(dataset_path),
            message=f"データセット '{request.dataset_name}' を作成しました"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=List[DatasetInfo])
async def list_datasets():
    """利用可能なデータセット一覧を取得"""
    try:
        datasets = []
        
        if DATASETS_BASE_PATH.exists():
            for dataset_dir in DATASETS_BASE_PATH.iterdir():
                if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
                    # データセット構造の確認（train/val/testの存在をチェック）
                    has_structure = (
                        (dataset_dir / 'train' / 'images').exists() or
                        (dataset_dir / 'val' / 'images').exists() or
                        (dataset_dir / 'test' / 'images').exists()
                    )
                    
                    datasets.append(DatasetInfo(
                        name=dataset_dir.name,
                        path=str(dataset_dir),
                        exists=True  # フォルダが存在すれば選択可能にする
                    ))
        
        # ソート（名前順）
        datasets.sort(key=lambda x: x.name)
        
        return datasets
    
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/split", response_model=DatasetSplitResponse)
async def split_dataset(request: DatasetSplitRequest):
    """画像データをスプリットしてデータセットに追加"""
    try:
        # 比率の検証
        ratio_sum = request.train_ratio + request.val_ratio + request.test_ratio
        if abs(ratio_sum - 1.0) > 0.001:
            raise HTTPException(
                status_code=400, 
                detail=f"比率の合計が1.0になりません: {ratio_sum}"
            )
        
        # パスの構築
        base_path = Path(request.base_path)
        images_dir = base_path / request.class_name / "images"
        labels_dir = base_path / request.class_name / "labels"
        
        # ディレクトリの検証
        if not images_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"画像ディレクトリが見つかりません: {images_dir}"
            )
        
        # ラベル付き画像を取得
        labeled_images = get_labeled_images(images_dir, labels_dir, request.class_name)
        if not labeled_images:
            # より詳細なエラーメッセージ
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = list(images_dir.iterdir()) if images_dir.exists() else []
            label_files = list(labels_dir.iterdir()) if labels_dir.exists() else []
            
            image_count = len([f for f in image_files if f.suffix.lower() in image_extensions])
            label_count = len([f for f in label_files if f.suffix.lower() == '.txt'])
            
            if image_count == 0:
                detail = f"画像ファイルが見つかりません: {images_dir}"
            elif not labels_dir.exists():
                detail = f"ラベルディレクトリが見つかりません: {labels_dir}"
            elif label_count == 0:
                detail = f"ラベルファイルが見つかりません: {labels_dir}"
            else:
                detail = f"ラベル付き画像が見つかりません。画像: {image_count}枚, ラベル: {label_count}個（ペアなし）"
            
            raise HTTPException(status_code=400, detail=detail)
        
        # 画像枚数制限が指定されている場合は制限する
        if request.max_images and len(labeled_images) > request.max_images:
            # ランダムに選択
            np.random.seed(request.random_seed)
            indices = np.random.choice(len(labeled_images), request.max_images, replace=False)
            labeled_images = [labeled_images[i] for i in indices]
            logger.info(f"画像数を {request.max_images} 枚に制限しました")
        
        # データセットパスを取得
        dataset_path = get_dataset_path(request.destination_dataset)
        
        # 新規作成または既存確認
        if request.create_new:
            if dataset_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"データセット '{request.destination_dataset}' は既に存在します"
                )
        else:
            if not dataset_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"データセット '{request.destination_dataset}' が見つかりません"
                )
        
        # データセット構造を作成（新規作成時は空のdataset.yamlも作成）
        paths = create_dataset_structure(dataset_path, create_yaml=request.create_new)
        
        # 既存ファイルを収集
        existing_files = set()
        for split in ['train', 'val', 'test']:
            images_dir = paths[f'{split}_images']
            if images_dir.exists():
                existing_files.update(f.name for f in images_dir.iterdir())
        
        # データを分割
        train_items, val_items, test_items = split_data(
            labeled_images, 
            request.train_ratio, 
            request.val_ratio, 
            request.test_ratio,
            request.random_seed
        )
        
        # ファイルをコピー
        train_copied, train_skipped = copy_with_rename(
            train_items, paths['train_images'], paths['train_labels'], existing_files
        )
        val_copied, val_skipped = copy_with_rename(
            val_items, paths['val_images'], paths['val_labels'], existing_files
        )
        test_copied, test_skipped = copy_with_rename(
            test_items, paths['test_images'], paths['test_labels'], existing_files
        )
        
        total_copied = train_copied + val_copied + test_copied
        total_skipped = train_skipped + val_skipped + test_skipped
        
        # dataset.yamlを作成/更新
        update_dataset_yaml(dataset_path, request.class_name)
        
        return DatasetSplitResponse(
            success=True,
            train_count=train_copied,
            val_count=val_copied,
            test_count=test_copied,
            skipped_count=total_skipped,
            total_processed=total_copied,
            train_percentage=(train_copied / total_copied * 100) if total_copied > 0 else 0,
            val_percentage=(val_copied / total_copied * 100) if total_copied > 0 else 0,
            test_percentage=(test_copied / total_copied * 100) if total_copied > 0 else 0
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to split dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete-class", response_model=DeleteClassResponse)
async def delete_class(request: DeleteClassRequest):
    """指定クラスのファイルを削除"""
    try:
        dataset_path = get_dataset_path(request.dataset_name)
        
        if not dataset_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"データセット '{request.dataset_name}' が見つかりません"
            )
        
        deletion_counts = {
            'train': 0,
            'val': 0,
            'test': 0
        }
        
        # 各スプリットでファイルを削除
        for split in ['train', 'val', 'test']:
            images_dir = dataset_path / split / 'images'
            labels_dir = dataset_path / split / 'labels'
            
            if images_dir.exists():
                # 対象ファイルを検索
                pattern = f"{request.class_name}_*"
                
                # 画像ファイルを削除
                for image_file in images_dir.glob(pattern):
                    if image_file.is_file():
                        # 対応するラベルファイルも削除
                        label_file = labels_dir / f"{image_file.stem}.txt"
                        
                        image_file.unlink()
                        if label_file.exists():
                            label_file.unlink()
                        
                        deletion_counts[split] += 1
        
        # dataset.yamlからクラスを削除
        update_dataset_yaml_remove_class(dataset_path, request.class_name)
        
        total_deleted = sum(deletion_counts.values())
        
        return DeleteClassResponse(
            success=True,
            train_deleted=deletion_counts['train'],
            val_deleted=deletion_counts['val'],
            test_deleted=deletion_counts['test'],
            total_deleted=total_deleted
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete class: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def update_dataset_yaml(dataset_path: Path, new_class: str):
    """dataset.yamlを更新（クラス追加）"""
    yaml_path = dataset_path / 'dataset.yaml'
    
    # 既存のクラスを読み込み
    classes = []
    if yaml_path.exists():
        with open(yaml_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 簡易的なパース（より堅牢にするならyamlライブラリを使用）
            for line in content.split('\n'):
                if line.startswith('names:'):
                    # names: ['class1', 'class2'] の形式を想定
                    import ast
                    try:
                        classes = ast.literal_eval(line.split(':', 1)[1].strip())
                    except:
                        classes = []
                    break
    
    # 新しいクラスを追加
    if new_class not in classes:
        classes.append(new_class)
    
    # YAMLを書き込み
    yaml_content = f"""# YOLOv8 Dataset Configuration
path: {dataset_path.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
nc: {len(classes)}
names: {classes}
"""
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)


def update_dataset_yaml_remove_class(dataset_path: Path, class_to_remove: str):
    """dataset.yamlを更新（クラス削除）"""
    yaml_path = dataset_path / 'dataset.yaml'
    
    if not yaml_path.exists():
        return
    
    # 既存のクラスを読み込み
    classes = []
    with open(yaml_path, 'r', encoding='utf-8') as f:
        content = f.read()
        for line in content.split('\n'):
            if line.startswith('names:'):
                import ast
                try:
                    classes = ast.literal_eval(line.split(':', 1)[1].strip())
                except:
                    classes = []
                break
    
    # クラスを削除
    if class_to_remove in classes:
        classes.remove(class_to_remove)
    
    # YAMLを書き込み
    yaml_content = f"""# YOLOv8 Dataset Configuration
path: {dataset_path.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
nc: {len(classes)}
names: {classes}
"""
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)


@router.post("/delete-dataset", response_model=DeleteDatasetResponse)
async def delete_dataset(request: DeleteDatasetRequest):
    """データセット全体を削除"""
    try:
        dataset_path = get_dataset_path(request.dataset_name)
        
        if not dataset_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"データセット '{request.dataset_name}' が見つかりません"
            )
        
        if not request.confirm:
            raise HTTPException(
                status_code=400,
                detail="削除を実行するには confirm=true を設定してください"
            )
        
        # データセットディレクトリを削除
        shutil.rmtree(dataset_path)
        logger.info(f"Deleted dataset: {request.dataset_name} at {dataset_path}")
        
        return DeleteDatasetResponse(
            success=True,
            deleted_path=str(dataset_path),
            message=f"データセット '{request.dataset_name}' を削除しました"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))