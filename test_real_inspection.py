#!/usr/bin/env python3
"""
実際の画像検査テスト
"""

import asyncio
import os
import sys
from datetime import datetime


async def test_real_inspection():
    """実際の画像を検査"""
    
    from src.core.inspection_engine import get_inspection_engine
    from src.models.inspection_schemas import InspectionRequest, InspectionMode
    
    # 設定
    seller_id = "A3OBH97MEO1982"
    max_items = 5
    
    print(f"\n=== 実際の画像検査テスト ===")
    print(f"セラーID: {seller_id}")
    print(f"最大検査数: {max_items}")
    print(f"画像パス: /mnt/c/03_amazon_images/{seller_id}/")
    
    # パスの確認
    base_path = f"/mnt/c/03_amazon_images/{seller_id}"
    if not os.path.exists(base_path):
        print(f"\nエラー: パスが存在しません: {base_path}")
        return
    
    # サブフォルダの確認
    print(f"\nサブフォルダ:")
    subfolders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    for folder in subfolders[:5]:
        folder_path = os.path.join(base_path, folder)
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  {folder}: {len(images)} images")
        if images:
            print(f"    例: {images[0]}")
    
    # 検査エンジンを初期化
    engine = get_inspection_engine()
    
    # リクエストを作成
    request = InspectionRequest(
        mode=InspectionMode.INDIVIDUAL,
        model_name="bandai",  # 実在するモデル名に変更
        device_mode="cpu",
        seller_id=seller_id,
        max_items=max_items,
        gpu_load_rate=0.8,
        confidence_threshold=0.5,
        max_detections=10
    )
    
    # バッチを開始
    print(f"\n検査を開始します...")
    batch_id = await engine.process_inspection_batch(request)
    print(f"バッチID: {batch_id}")
    
    # 進捗を監視
    print("\n進捗:")
    for i in range(30):  # 最大30秒待機
        status = engine.get_batch_status(batch_id)
        if status:
            print(f"\r  {status.status}: {status.items_processed}/{status.items_total} "
                  f"({status.progress:.1f}%)", end="")
            
            if status.status in ["completed", "failed", "cancelled"]:
                print()  # 改行
                break
        
        await asyncio.sleep(1)
    
    # 結果を取得
    print("\n\n結果:")
    batch_result = engine.get_batch_results(batch_id)
    
    if batch_result:
        print(f"総アイテム数: {batch_result.total_items}")
        print(f"成功: {batch_result.successful_items}")
        print(f"失敗: {batch_result.failed_items}")
        
        print("\n詳細結果:")
        for i, result in enumerate(batch_result.results[:10]):  # 最初の10件
            print(f"\n[{i+1}] ASIN: {result.asin}")
            print(f"    画像パス: {result.image_path}")
            print(f"    検出: {'あり' if result.detected else 'なし'}")
            if result.detected:
                print(f"    ラベル: {', '.join(result.labels)}")
                print(f"    信頼度: {[f'{s:.1%}' for s in result.confidence_scores]}")
            if result.error:
                print(f"    エラー: {result.error}")
    else:
        print("結果が取得できませんでした")


if __name__ == "__main__":
    asyncio.run(test_real_inspection())