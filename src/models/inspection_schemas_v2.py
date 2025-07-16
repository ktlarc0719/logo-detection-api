"""
画像検査機能用のスキーマ定義 V2
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class InspectionMode(str, Enum):
    """検査モード"""
    SELLER = "seller"  # セラーID指定
    PATH = "path"     # 絶対パス指定


class InspectionRequest(BaseModel):
    """画像検査リクエスト"""
    mode: InspectionMode
    model_name: str = Field(..., description="使用するモデル名")
    
    # モード別パラメータ
    sellers: List[str] = Field(default_factory=list, description="セラーIDリスト")
    base_path: Optional[str] = Field(None, description="基準パス（パスモード用）")
    include_subdirs: bool = Field(True, description="サブディレクトリを含むか")
    
    # 検査対象選択
    only_uninspected: bool = Field(True, description="未検査のみを対象とするか")
    
    # 共通パラメータ
    confidence_threshold: float = Field(0.5, ge=0.1, le=1.0)
    max_detections: int = Field(10, ge=1, le=100)
    max_items: Optional[int] = Field(None, ge=1)  # None = 全件処理
    
    # デバイス設定
    device_mode: Optional[str] = Field("cpu", description="デバイスモード (cpu/gpu)")
    gpu_load_rate: Optional[float] = Field(0.8, ge=0.1, le=1.0, description="GPU負荷率")


class InspectionItem(BaseModel):
    """検査対象アイテム"""
    item_id: str
    seller_id: Optional[str] = None
    asin: Optional[str] = None
    image_path: str


class InspectionResult(BaseModel):
    """検査結果"""
    batch_id: Optional[str] = None
    item_id: str
    seller_id: Optional[str] = None
    asin: Optional[str] = None
    image_path: str
    detected: bool
    detections: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_scores: List[float] = Field(default_factory=list)
    labels: List[str] = Field(default_factory=list)
    model_used: str
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None


class InspectionStatus(BaseModel):
    """検査ステータス"""
    batch_id: str
    status: str  # pending, running, completed, failed, cancelled
    total_items: int
    processed_items: int = 0
    detected_items: int = 0
    failed_items: int = 0
    progress: float = Field(0.0, ge=0.0, le=100.0)
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # 検査パラメータ
    mode: InspectionMode
    seller_ids: List[str] = Field(default_factory=list)
    base_path: Optional[str] = None
    
    # エラー情報
    error_message: Optional[str] = None


class InspectionBatchResult(BaseModel):
    """バッチ検査結果"""
    batch_id: str
    status: InspectionStatus
    results: List[InspectionResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)


class InspectionDashboard(BaseModel):
    """管理UI用ダッシュボードデータ"""
    active_batches: List[InspectionStatus]
    available_models: List[Dict[str, Any]]
    gpu_status: Dict[str, Any]
    recent_results: List[InspectionBatchResult]
    statistics: Dict[str, Any]