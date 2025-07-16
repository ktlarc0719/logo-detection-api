#!/usr/bin/env python3
"""
検査UIのデバッグスクリプト
"""

import requests

url = "http://localhost:8000/ui/inspection"

try:
    response = requests.get(url)
    content = response.text
    
    # デバイスモードが含まれているか確認
    if "デバイスモード" in content:
        print("✓ デバイスモードが見つかりました")
        
        # デバイスモード周辺のコンテキストを表示
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "デバイスモード" in line:
                print(f"\n行 {i}: {line.strip()}")
                # 前後5行を表示
                for j in range(max(0, i-5), min(len(lines), i+6)):
                    print(f"  {j}: {lines[j].strip()[:80]}...")
    else:
        print("✗ デバイスモードが見つかりません")
        
    # Last Updatedが含まれているか確認
    if "Last Updated: 2025-01-13" in content:
        print("\n✓ 最新バージョンが提供されています")
    else:
        print("\n✗ 古いバージョンが提供されている可能性があります")
        
    # CPUモードのラジオボタンを確認
    if 'name="device_mode" value="cpu"' in content:
        print("✓ CPUモードのラジオボタンが見つかりました")
    else:
        print("✗ CPUモードのラジオボタンが見つかりません")

except Exception as e:
    print(f"エラー: {e}")