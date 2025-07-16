"""
画像検査機能用のスキーマ定義
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class InspectionMode(str, Enum):
    """検査実行モード"""
    INDIVIDUAL = "individual"  # 個別指定モード
    BULK = "bulk"              # 全件実行モード


class InspectionRequest(BaseModel):
    """画像検査リクエスト"""
    mode: InspectionMode
    model_name: str = Field(..., description="使用するモデル名")
    gpu_load_rate: float = Field(0.8, ge=0.1, le=1.0, description="GPU負荷率（0.1-1.0）")
    device_mode: str = Field("cpu", description="デバイスモード (cpu/gpu)")
    
    # 個別指定モード用
    seller_id: Optional[str] = Field(None, description="セラーID（個別指定時）")
    user_id: Optional[str] = Field(None, description="ユーザーID（個別指定時）")
    
    # 処理上限
    max_items: Optional[int] = Field(None, ge=1, description="処理上限数")
    
    # 全件実行モード用
    process_all: bool = Field(True, description="全件処理フラグ")
    
    # 検査パラメータ
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_detections: int = Field(10, ge=1, le=100)


class InspectionItem(BaseModel):
    """検査対象アイテム"""
    seller_id: str
    asin: str
    user_id: Optional[str] = None
    image_path: str
    status: str = "pending"  # pending, processing, completed, failed
    created_at: datetime = Field(default_factory=datetime.utcnow)


class InspectionResult(BaseModel):
    """個別の検査結果"""
    item_id: str
    seller_id: str
    asin: str
    user_id: Optional[str] = None
    image_path: str
    
    # 検査結果
    detected: bool
    detections: List[Dict[str, Any]] = []
    confidence_scores: List[float] = []
    labels: List[str] = []
    
    # メタデータ
    model_used: str
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None


class InspectionBatchResult(BaseModel):
    """バッチ検査結果"""
    batch_id: str
    mode: InspectionMode
    
    # 統計情報
    total_items: int
    processed_items: int
    successful_items: int
    failed_items: int
    
    # 結果
    results: List[InspectionResult]
    
    # 実行情報
    model_used: str
    gpu_load_rate: float
    start_time: datetime
    end_time: Optional[datetime] = None
    total_processing_time: Optional[float] = None
    

class InspectionStatus(BaseModel):
    """検査ステータス"""
    batch_id: str
    status: str  # running, completed, failed, cancelled
    progress: float = Field(0.0, ge=0.0, le=100.0)
    current_item: Optional[str] = None
    items_processed: int = 0
    items_total: int = 0
    estimated_time_remaining: Optional[float] = None
    errors: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # 追加フィールド
    model_name: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    elapsed_time: Optional[float] = None  # seconds
    images_per_hour: Optional[float] = None


class InspectionDashboard(BaseModel):
    """管理UI用ダッシュボードデータ"""
    active_batches: List[InspectionStatus]
    available_models: List[Dict[str, Any]]
    gpu_status: Dict[str, Any]
    recent_results: List[InspectionBatchResult]
    statistics: Dict[str, Any]