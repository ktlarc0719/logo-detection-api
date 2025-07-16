#!/usr/bin/env python3
"""
セラーIDモードの直接テスト（サーバーを立ち上げずにテスト）
"""

import asyncio
import os
from datetime import datetime
import torch


async def test_seller_inspection():
    """セラーID検査を直接テスト"""
    
    # インポート
    from src.core.inspection_engine import get_inspection_engine
    from src.models.inspection_schemas import InspectionRequest, InspectionMode
    
    # 設定
    seller_id = "A3OBH97MEO1982"
    max_items = 5
    device_mode = "cpu"
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting direct test")
    print(f"Seller ID: {seller_id}")
    print(f"Device mode: {device_mode}")
    print(f"Max items: {max_items}")
    print(f"Base path: /mnt/c/03_amazon_images")
    
    # デバイス情報を表示
    print(f"\nDevice Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # 検査エンジンを初期化
    engine = get_inspection_engine()
    
    # パスの存在確認
    seller_path = os.path.join("/mnt/c/03_amazon_images", seller_id)
    print(f"\nChecking seller path: {seller_path}")
    print(f"Path exists: {os.path.exists(seller_path)}")
    
    if os.path.exists(seller_path):
        # サブフォルダを確認
        subfolders = [d for d in os.listdir(seller_path) if os.path.isdir(os.path.join(seller_path, d))]
        print(f"Subfolders found: {len(subfolders)}")
        print(f"First 5 subfolders: {subfolders[:5]}")
        
        # 最初のサブフォルダの画像を確認
        if subfolders:
            first_subfolder = os.path.join(seller_path, subfolders[0])
            images = [f for f in os.listdir(first_subfolder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"\nImages in subfolder '{subfolders[0]}': {len(images)}")
            if images:
                print(f"First 3 images: {images[:3]}")
    
    # リクエストを作成
    request = InspectionRequest(
        mode=InspectionMode.INDIVIDUAL,
        model_name="logo_detector_v30",
        device_mode=device_mode,
        seller_id=seller_id,
        max_items=max_items,
        gpu_load_rate=0.8,
        confidence_threshold=0.5,
        max_detections=10
    )
    
    # デバイスモードを設定
    engine.set_device_mode(device_mode)
    
    # 画像を取得
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Getting images...")
    items = await engine._get_inspection_items(request)
    print(f"Found {len(items)} images")
    
    if items:
        print("\nFirst 3 items:")
        for i, item in enumerate(items[:3]):
            print(f"  [{i+1}] ASIN: {item.asin}")
            print(f"       Path: {item.image_path}")
            print(f"       Exists: {os.path.exists(item.image_path)}")
    
    # 検査を実行（最初の3つのみ）
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running inspection...")
    for i, item in enumerate(items[:3]):
        print(f"\n  Processing [{i+1}/{min(3, len(items))}]: {item.asin}")
        
        try:
            result = await engine._inspect_single_item(
                item,
                request.confidence_threshold,
                request.max_detections
            )
            
            print(f"    Detected: {result.detected}")
            if result.detected:
                print(f"    Labels: {result.labels}")
                print(f"    Confidences: {[f'{c:.2f}' for c in result.confidence_scores]}")
            print(f"    Processing time: {result.processing_time:.3f}s")
            
            if result.error:
                print(f"    Error: {result.error}")
                
        except Exception as e:
            print(f"    Error: {e}")
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Test completed")


if __name__ == "__main__":
    asyncio.run(test_seller_inspection())