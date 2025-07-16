#!/usr/bin/env python3
"""
新しい検査機能のテストスクリプト
"""

import asyncio
import requests
import json
import time

# APIのベースURL
API_BASE_URL = "http://localhost:8000/api/v1"

def test_queue_status():
    """キューステータスの確認"""
    print("\n=== キューステータスの確認 ===")
    response = requests.get(f"{API_BASE_URL}/inspection/queue/status")
    if response.status_code == 200:
        data = response.json()
        print(f"キュー長: {data['queue_length']}")
        print(f"処理中: {data['processing']}")
        print(f"待機中アイテム数: {len(data['pending_items'])}")
        return data
    else:
        print(f"エラー: {response.status_code}")
        return None

def test_start_inspection(seller_id="A3OBH97MEO1982", only_uninspected=True):
    """検査開始のテスト"""
    print(f"\n=== 検査開始のテスト (Seller: {seller_id}, 未検査のみ: {only_uninspected}) ===")
    
    # リクエストデータ
    request_data = {
        "mode": "seller",
        "model_name": "general",
        "sellers": [seller_id],
        "only_uninspected": only_uninspected,
        "confidence_threshold": 0.5,
        "max_detections": 10,
        "device_mode": "cpu",
        "gpu_load_rate": 0.8,
        "max_items": 10  # テスト用に少なめに設定
    }
    
    print(f"リクエスト: {json.dumps(request_data, indent=2)}")
    
    response = requests.post(
        f"{API_BASE_URL}/inspection/start",
        json=request_data
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"成功: キューID = {data['queue_id']}")
        return data['queue_id']
    else:
        print(f"エラー: {response.status_code}")
        print(response.json())
        return None

def test_check_queue_item(queue_id):
    """キューアイテムの状態確認"""
    print(f"\n=== キューアイテムの確認 (Queue ID: {queue_id}) ===")
    response = requests.get(f"{API_BASE_URL}/inspection/queue/{queue_id}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"ステータス: {data['status']}")
        print(f"位置: {data.get('position', 'N/A')}")
        print(f"バッチID: {data.get('batch_id', 'N/A')}")
        return data
    else:
        print(f"エラー: {response.status_code}")
        return None

def test_get_batch_status(batch_id):
    """バッチステータスの確認"""
    print(f"\n=== バッチステータスの確認 (Batch ID: {batch_id}) ===")
    response = requests.get(f"{API_BASE_URL}/inspection/status/{batch_id}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"ステータス: {data['status']}")
        print(f"進捗: {data['processed_items']}/{data['total_items']} ({data['progress']:.1f}%)")
        print(f"検出: {data['detected_items']}")
        return data
    else:
        print(f"エラー: {response.status_code}")
        return None

def test_cancel_queue_item(queue_id):
    """キューアイテムのキャンセルテスト"""
    print(f"\n=== キューアイテムのキャンセル (Queue ID: {queue_id}) ===")
    response = requests.delete(f"{API_BASE_URL}/inspection/queue/{queue_id}")
    
    if response.status_code == 200:
        print("キャンセル成功")
        return True
    else:
        print(f"エラー: {response.status_code}")
        print(response.json())
        return False

async def main():
    """メインテスト関数"""
    print("新しい検査機能のテスト開始")
    
    # 1. 初期のキューステータス確認
    test_queue_status()
    
    # 2. 検査を開始（未検査のみ）
    queue_id1 = test_start_inspection(seller_id="A3OBH97MEO1982", only_uninspected=True)
    
    # 3. もう一つ検査を追加（全検査）
    queue_id2 = test_start_inspection(seller_id="A3OBH97MEO1982", only_uninspected=False)
    
    if queue_id1 and queue_id2:
        # 4. キューステータスを再確認
        time.sleep(1)
        test_queue_status()
        
        # 5. 各キューアイテムの状態確認
        item1 = test_check_queue_item(queue_id1)
        item2 = test_check_queue_item(queue_id2)
        
        # 6. 2番目のキューアイテムをキャンセル
        if item2 and item2['status'] == 'pending':
            test_cancel_queue_item(queue_id2)
            time.sleep(1)
            test_queue_status()
        
        # 7. 処理完了を待つ
        print("\n=== 処理完了を待機中... ===")
        max_wait = 60  # 最大60秒待機
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            item = test_check_queue_item(queue_id1)
            if item and item['status'] == 'completed':
                print("\n処理完了！")
                if item.get('batch_id'):
                    test_get_batch_status(item['batch_id'])
                break
            elif item and item['status'] == 'failed':
                print("\n処理失敗")
                print(f"エラー: {item.get('error_message', 'Unknown error')}")
                break
            
            time.sleep(5)
        else:
            print("\nタイムアウト：処理が完了しませんでした")
    
    print("\nテスト完了")

if __name__ == "__main__":
    asyncio.run(main())