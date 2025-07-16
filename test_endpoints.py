#!/usr/bin/env python3
"""
エンドポイントのテストスクリプト
"""

import requests
import json

API_BASE_URL = "http://localhost:8000/api/v1"

def test_endpoint(name, url, method="GET", data=None):
    """エンドポイントをテスト"""
    print(f"\n=== Testing {name} ===")
    print(f"URL: {url}")
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            print(f"Unsupported method: {method}")
            return
            
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Success!")
            print(f"Response: {json.dumps(response.json(), indent=2)[:500]}...")
        else:
            print("Error!")
            print(f"Response: {response.text[:500]}")
            
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")

def main():
    """メイン関数"""
    print("Testing Inspection API Endpoints")
    
    # 1. Models endpoint
    test_endpoint(
        "Get Available Models",
        f"{API_BASE_URL}/inspection/models"
    )
    
    # 2. Device Info endpoint
    test_endpoint(
        "Get Device Info",
        f"{API_BASE_URL}/inspection/device-info"
    )
    
    # 3. Active batches endpoint
    test_endpoint(
        "Get Active Batches",
        f"{API_BASE_URL}/inspection/active"
    )
    
    # 4. Dashboard endpoint
    test_endpoint(
        "Get Dashboard",
        f"{API_BASE_URL}/inspection/dashboard"
    )
    
    # 5. Queue status endpoint
    test_endpoint(
        "Get Queue Status",
        f"{API_BASE_URL}/inspection/queue/status"
    )

if __name__ == "__main__":
    main()