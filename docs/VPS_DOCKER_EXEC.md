# VPS上でDockerコンテナ内のコード実行方法

## 1. docker exec を使った直接実行

```bash
# Pythonコードを直接実行
docker exec logo-detection-api python -c "print('Hello from container')"

# スクリプトファイルを実行
docker exec logo-detection-api python /app/test_script.py

# インタラクティブなPythonシェル
docker exec -it logo-detection-api python

# bashシェルに入る
docker exec -it logo-detection-api bash
```

## 2. APIエンドポイント経由での実行

現在のAPIには `/api/v1/detect/batch` などのエンドポイントがあるので：

```bash
# バッチ処理を実行
curl -X POST http://localhost:8000/api/v1/detect/batch \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com/image1.jpg"]}'
```

## 3. カスタムスクリプトをコンテナにコピーして実行

```bash
# ローカルのスクリプトをコンテナにコピー
docker cp test_memory.py logo-detection-api:/app/

# コピーしたスクリプトを実行
docker exec logo-detection-api python /app/test_memory.py
```

## 4. メモリ使用量の監視

```bash
# リアルタイムでメモリ使用量を確認
docker stats logo-detection-api

# メモリ使用量の詳細
docker exec logo-detection-api cat /proc/meminfo

# Pythonプロセスのメモリ使用量
docker exec logo-detection-api ps aux | grep python
```

## 5. テスト用のPythonコード例

```bash
# メモリ使用量をテストするコード
docker exec logo-detection-api python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"

# モデルのロード状態を確認
docker exec logo-detection-api python -c "
from app.services.logo_detector import LogoDetector
detector = LogoDetector()
print(f'Model loaded: {detector.model is not None}')
"
```

## 6. ログの確認

```bash
# コンテナのログを確認
docker logs logo-detection-api

# リアルタイムでログを監視
docker logs -f logo-detection-api

# 最新の50行
docker logs --tail 50 logo-detection-api
```

## 7. デバッグモードでの実行

```bash
# 環境変数を設定して実行
docker exec -e DEBUG=true logo-detection-api python /app/debug_script.py

# 一時的なテストコンテナを起動
docker run --rm -it \
  --memory 1.5g \
  --memory-reservation 1g \
  -v $(pwd)/test_scripts:/test \
  kentatsujikawadev/logo-detection-api:latest \
  python /test/memory_test.py
```