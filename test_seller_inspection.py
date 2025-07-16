#!/usr/bin/env python3
"""
セラーID画像検査機能のテストスクリプト
"""

import requests
import json
import time
import sys
from datetime import datetime


def test_seller_inspection(seller_id: str, max_items: int = 10, device_mode: str = "cpu"):
    """セラーID検査をテスト"""
    
    base_url = "http://localhost:8000"
    
    # 1. 検査バッチを開始
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting inspection for seller: {seller_id}")
    print(f"Device mode: {device_mode}, Max items: {max_items}")
    
    request_data = {
        "mode": "individual",
        "model_name": "logo_detector_v30",
        "device_mode": device_mode,
        "seller_id": seller_id,
        "max_items": max_items,
        "gpu_load_rate": 0.8,
        "confidence_threshold": 0.5,
        "max_detections": 10
    }
    
    response = requests.post(f"{base_url}/api/inspection/batch", json=request_data)
    
    if response.status_code != 200:
        print(f"Error starting batch: {response.text}")
        return
    
    batch_id = response.json()["batch_id"]
    print(f"Batch ID: {batch_id}")
    
    # 2. 進捗を監視
    print("\nMonitoring progress...")
    last_progress = -1
    
    while True:
        response = requests.get(f"{base_url}/api/inspection/batch/{batch_id}/status")
        
        if response.status_code != 200:
            print(f"Error getting status: {response.text}")
            break
        
        status = response.json()
        current_progress = status["progress"]
        
        # 進捗が更新された場合のみ表示
        if current_progress != last_progress:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Status: {status['status']}, "
                  f"Progress: {current_progress:.1f}%, "
                  f"Processed: {status['items_processed']}/{status['items_total']}")
            
            if status.get("current_item"):
                print(f"  Current item: {status['current_item']}")
            
            last_progress = current_progress
        
        # エラーがあれば表示
        if status.get("errors"):
            print(f"  Errors: {status['errors']}")
        
        # 完了したら終了
        if status["status"] in ["completed", "failed", "cancelled"]:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Batch {status['status']}")
            break
        
        time.sleep(1)
    
    # 3. 結果を取得
    print("\nGetting results...")
    response = requests.get(f"{base_url}/api/inspection/batch/{batch_id}/results")
    
    if response.status_code == 200:
        results = response.json()
        print(f"\nTotal items: {results['total_items']}")
        print(f"Successful: {results['successful_items']}")
        print(f"Failed: {results['failed_items']}")
        
        # 検出結果のサマリー
        if results.get("results"):
            detected_count = sum(1 for r in results["results"] if r["detected"])
            print(f"\nDetections found in {detected_count}/{len(results['results'])} images")
            
            # 各結果を表示
            print("\nDetailed results:")
            for i, result in enumerate(results["results"][:5]):  # 最初の5件のみ表示
                print(f"\n  [{i+1}] ASIN: {result['asin']}")
                print(f"  Path: {result['image_path']}")
                print(f"  Detected: {result['detected']}")
                
                if result["detected"]:
                    print(f"  Labels: {', '.join(result['labels'])}")
                    print(f"  Confidences: {[f'{c:.2f}' for c in result['confidence_scores']]}")
                
                if result.get("error"):
                    print(f"  Error: {result['error']}")
            
            if len(results["results"]) > 5:
                print(f"\n  ... and {len(results['results']) - 5} more results")
    else:
        print(f"Error getting results: {response.text}")


def test_device_info():
    """デバイス情報を取得"""
    
    base_url = "http://localhost:8000"
    
    print("\nGetting device information...")
    response = requests.get(f"{base_url}/api/inspection/device-info")
    
    if response.status_code == 200:
        info = response.json()
        print(f"CPU available: Yes")
        print(f"GPU available: {info['gpu_status']['available']}")
        
        if info['gpu_status']['available']:
            print(f"GPU device: {info['gpu_status']['device_name']}")
            print(f"GPU memory allocated: {info['gpu_status']['memory_allocated']:.2f} GB")
    else:
        print(f"Error getting device info: {response.text}")


def main():
    """メイン関数"""
    
    if len(sys.argv) < 2:
        print("Usage: python test_seller_inspection.py <seller_id> [max_items] [device_mode]")
        print("Example: python test_seller_inspection.py A3OBH97MEO1982 20 cpu")
        sys.exit(1)
    
    seller_id = sys.argv[1]
    max_items = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    device_mode = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    
    # デバイス情報を表示
    test_device_info()
    
    # セラー検査を実行
    test_seller_inspection(seller_id, max_items, device_mode)


if __name__ == "__main__":
    main()