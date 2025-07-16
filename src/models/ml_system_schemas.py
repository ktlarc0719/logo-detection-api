"""
機械学習システム用のスキーマ定義
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class DatasetType(str, Enum):
    """データセットタイプ"""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class TrainingMode(str, Enum):
    """学習モード"""
    FULL = "full"  # ゼロからの学習
    TRANSFER = "transfer"  # 転移学習
    CONTINUE = "continue"  # 継続学習


class ModelArchitecture(str, Enum):
    """利用可能なモデルアーキテクチャ"""
    YOLOV8N = "yolov8n"
    YOLOV8S = "yolov8s"
    YOLOV8M = "yolov8m"
    YOLOV8L = "yolov8l"
    YOLOV8X = "yolov8x"


class DatasetValidationRequest(BaseModel):
    """データセット検証リクエスト"""
    dataset_path: str = Field(..., description="データセットのパス")
    check_images: bool = Field(True, description="画像ファイルの存在確認")
    check_labels: bool = Field(True, description="ラベルファイルの確認")
    validate_format: bool = Field(True, description="フォーマットの検証")


class DatasetValidationResult(BaseModel):
    """データセット検証結果"""
    dataset_path: str
    valid: bool
    total_images: int
    total_labels: int
    
    # 各分割の情報
    train_info: Optional[Dict[str, Any]] = None
    val_info: Optional[Dict[str, Any]] = None
    test_info: Optional[Dict[str, Any]] = None
    
    # クラス情報
    classes: List[str]
    class_distribution: Dict[str, Dict[str, int]]  # {split: {class: count}}
    
    # 問題点
    errors: List[str] = []
    warnings: List[str] = []
    
    # 統計情報
    statistics: Dict[str, Any] = {}


class TrainingRequest(BaseModel):
    """学習リクエスト"""
    mode: TrainingMode
    dataset_path: str
    model_architecture: ModelArchitecture = ModelArchitecture.YOLOV8S
    base_model_path: Optional[str] = None  # 継続学習時のベースモデル
    
    # ハイパーパラメータ
    epochs: int = Field(100, ge=1, le=1000)
    batch_size: int = Field(16, ge=1, le=128)
    learning_rate: float = Field(0.01, ge=0.0001, le=0.1)
    imgsz: int = Field(640, ge=320, le=1280)
    
    # 拡張設定
    augmentation_params: Dict[str, Any] = {}
    optimizer: str = "SGD"
    patience: int = Field(50, ge=10, le=200)
    
    # その他
    project_name: str = Field("ml_training", description="プロジェクト名")
    experiment_name: Optional[str] = None
    save_period: int = Field(10, description="何エポックごとに保存するか")
    device: Optional[str] = None  # "0", "cpu", etc.


class TrainingStatus(BaseModel):
    """学習ステータス"""
    training_id: str
    status: Literal["initializing", "running", "completed", "failed", "cancelled"]
    mode: TrainingMode
    
    # 進捗情報
    current_epoch: int = 0
    total_epochs: int
    progress: float = Field(0.0, ge=0.0, le=100.0)
    
    # メトリクス
    current_metrics: Dict[str, float] = {}
    best_metrics: Dict[str, float] = {}
    
    # 時間情報
    start_time: datetime
    estimated_time_remaining: Optional[float] = None
    
    # エラー情報
    errors: List[str] = []
    
    # 出力パス
    output_dir: Optional[str] = None
    best_model_path: Optional[str] = None
    last_model_path: Optional[str] = None


class ModelVisualizationRequest(BaseModel):
    """モデル可視化リクエスト"""
    model_path: str
    include_confusion_matrix: bool = True
    include_pr_curve: bool = True
    include_f1_curve: bool = True
    include_training_history: bool = True
    include_class_metrics: bool = True


class ModelVisualizationResult(BaseModel):
    """モデル可視化結果"""
    model_path: str
    model_info: Dict[str, Any]
    
    # クラス別メトリクス
    class_metrics: Dict[str, Dict[str, float]]  # {class: {metric: value}}
    
    # 可視化データ
    confusion_matrix_path: Optional[str] = None
    pr_curve_path: Optional[str] = None
    f1_curve_path: Optional[str] = None
    training_history: Optional[Dict[str, List[float]]] = None
    
    # 全体メトリクス
    overall_metrics: Dict[str, float] = {}
    
    # その他の情報
    training_args: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ModelValidationRequest(BaseModel):
    """モデル検証リクエスト"""
    model_path: str
    test_dataset_path: str
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    save_predictions: bool = True
    save_confusion_matrix: bool = True
    analyze_errors: bool = True


class PredictionResult(BaseModel):
    """個別の予測結果"""
    image_path: str
    ground_truth: List[Dict[str, Any]]  # 正解ラベル
    predictions: List[Dict[str, Any]]  # 予測結果
    
    # メトリクス
    correct: bool
    confidence_scores: List[float]
    iou_scores: List[float] = []
    
    # エラー分析
    error_type: Optional[str] = None  # "false_positive", "false_negative", "misclassification"
    error_details: Optional[str] = None


class ModelValidationResult(BaseModel):
    """モデル検証結果"""
    model_path: str
    test_dataset_path: str
    
    # 全体メトリクス
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    map50: float
    map50_95: float
    
    # クラス別メトリクス
    class_metrics: Dict[str, Dict[str, float]]
    
    # 詳細結果
    total_images: int
    correct_predictions: int
    prediction_results: List[PredictionResult]
    
    # エラー分析
    error_analysis: Dict[str, Any] = {}
    common_errors: List[Dict[str, Any]] = []
    
    # 可視化パス
    confusion_matrix_path: Optional[str] = None
    error_samples_dir: Optional[str] = None
    
    # 実行情報
    confidence_threshold: float
    iou_threshold: float
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SystemStatus(BaseModel):
    """システム全体のステータス"""
    gpu_available: bool
    gpu_info: Optional[Dict[str, Any]] = None
    
    # リソース情報
    cpu_usage: float
    memory_usage: float
    disk_usage: Dict[str, float]
    
    # 実行中のタスク
    active_trainings: List[str] = []
    active_validations: List[str] = []
    
    # モデル情報
    available_models: List[Dict[str, Any]] = []
    available_datasets: List[Dict[str, Any]] = []