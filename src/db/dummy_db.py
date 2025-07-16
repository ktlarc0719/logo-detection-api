"""
ダミーデータベース実装（開発用）
"""

import random
import string
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio

from src.models.inspection_schemas import InspectionItem, InspectionResult


# ダミーデータ生成用の定数
SELLER_IDS = [f"SELLER_{i:03d}" for i in range(1, 21)]
USER_IDS = [f"USER_{i:04d}" for i in range(1, 101)]
BRAND_NAMES = ["BANDAI", "NINTENDO", "SONY", "POKEMON", "TOMICA"]


def generate_asin() -> str:
    """ランダムなASINを生成"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))


async def get_dummy_inspection_items(
    seller_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: Optional[int] = None
) -> List[InspectionItem]:
    """
    ダミーの検査対象アイテムを取得
    
    Args:
        seller_id: 特定のセラーIDでフィルタ
        user_id: 特定のユーザーIDでフィルタ
        limit: 取得上限数
        
    Returns:
        検査対象アイテムのリスト
    """
    # 非同期処理のシミュレーション
    await asyncio.sleep(0.1)
    
    items = []
    
    # 生成するアイテム数を決定
    if seller_id and user_id:
        # 個別指定の場合は少なめ
        total_items = random.randint(10, 50)
    else:
        # 全件の場合は多め
        total_items = random.randint(100, 500)
    
    # limitが指定されている場合は制限
    if limit:
        total_items = min(total_items, limit)
    
    # ダミーアイテムを生成
    for i in range(total_items):
        # seller_idが指定されていればそれを使用、なければランダム
        item_seller_id = seller_id if seller_id else random.choice(SELLER_IDS)
        
        # user_idが指定されていればそれを使用、なければランダム（30%の確率でNone）
        if user_id:
            item_user_id = user_id
        else:
            item_user_id = random.choice(USER_IDS) if random.random() > 0.3 else None
        
        item = InspectionItem(
            seller_id=item_seller_id,
            asin=generate_asin(),
            user_id=item_user_id,
            image_path="",  # エンジンで生成される
            status="pending",
            created_at=datetime.utcnow() - timedelta(days=random.randint(0, 30))
        )
        
        items.append(item)
    
    return items


async def save_inspection_result(result: InspectionResult) -> bool:
    """
    検査結果を保存（ダミー実装）
    
    Args:
        result: 検査結果
        
    Returns:
        成功した場合True
    """
    # 非同期処理のシミュレーション
    await asyncio.sleep(0.01)
    
    # ログ出力で保存をシミュレート
    print(f"Saving inspection result: ASIN={result.asin}, "
          f"Detected={result.detected}, Labels={result.labels}")
    
    # 実際のDB実装では、ここでINSERT文を実行
    # await db.execute(
    #     """
    #     INSERT INTO inspection_results 
    #     (item_id, seller_id, asin, user_id, detected, labels, confidence_scores, 
    #      model_used, processing_time, timestamp)
    #     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
    #     """,
    #     result.item_id, result.seller_id, result.asin, result.user_id,
    #     result.detected, result.labels, result.confidence_scores,
    #     result.model_used, result.processing_time, result.timestamp
    # )
    
    return True


async def get_inspection_statistics() -> Dict[str, Any]:
    """
    検査統計情報を取得（ダミー実装）
    
    Returns:
        統計情報の辞書
    """
    await asyncio.sleep(0.05)
    
    return {
        "total_inspections": random.randint(10000, 50000),
        "total_detections": random.randint(5000, 25000),
        "detection_rate": random.uniform(0.4, 0.6),
        "average_processing_time": random.uniform(0.05, 0.15),
        "inspections_today": random.randint(100, 1000),
        "inspections_this_week": random.randint(1000, 5000),
        "top_detected_brands": [
            {"brand": brand, "count": random.randint(100, 1000)}
            for brand in random.sample(BRAND_NAMES, 3)
        ],
        "by_seller": [
            {
                "seller_id": seller,
                "total": random.randint(100, 1000),
                "detected": random.randint(50, 500)
            }
            for seller in random.sample(SELLER_IDS, 5)
        ]
    }


async def get_recent_results(limit: int = 10) -> List[Dict[str, Any]]:
    """
    最近の検査結果を取得（ダミー実装）
    
    Args:
        limit: 取得件数
        
    Returns:
        最近の結果リスト
    """
    await asyncio.sleep(0.05)
    
    results = []
    for i in range(limit):
        detected = random.random() > 0.4
        results.append({
            "asin": generate_asin(),
            "seller_id": random.choice(SELLER_IDS),
            "detected": detected,
            "labels": random.sample(BRAND_NAMES, random.randint(0, 3)) if detected else [],
            "timestamp": datetime.utcnow() - timedelta(minutes=random.randint(0, 60)),
            "processing_time": random.uniform(0.05, 0.15)
        })
    
    return sorted(results, key=lambda x: x["timestamp"], reverse=True)