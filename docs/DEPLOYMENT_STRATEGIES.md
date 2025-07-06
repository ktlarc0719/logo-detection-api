# デプロイメント戦略ガイド

## 再ビルドが必要な場合と不要な場合

### 🔄 再ビルドが必要な場合

1. **Pythonコードの変更**
   - `src/` ディレクトリ内のファイル
   - APIエンドポイントの追加・変更
   - ビジネスロジックの変更

2. **依存関係の変更**
   - `requirements.txt` の更新
   - 新しいパッケージの追加

3. **Dockerfileの変更**
   - ベースイメージの変更
   - 環境変数のデフォルト値変更

4. **静的ファイルの変更**
   - モデルファイルの更新
   - 設定ファイルの構造変更

### ✅ 再ビルド不要な場合（再起動のみ）

1. **環境変数の値変更**
   - `MAX_CONCURRENT_DETECTIONS` などの調整
   - ログレベルの変更

2. **外部設定ファイル**
   - マウントされた設定ファイルの変更
   - ログ設定の調整

3. **ドキュメントのみの変更**
   - README.md
   - docs/ ディレクトリ

## デプロイ時のダウンタイム対策

### 現在の問題点
- コンテナ停止時に実行中のリクエストが強制終了
- 新しいコンテナ起動まで完全にサービス停止

### 推奨される対策

#### 1. グレースフルシャットダウンの実装

```python
# src/api/main.py に追加
import signal
import asyncio

shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    logger.info("Received shutdown signal, starting graceful shutdown...")
    shutdown_event.set()

signal.signal(signal.SIGTERM, signal_handler)

@app.on_event("shutdown")
async def shutdown():
    logger.info("Waiting for active requests to complete...")
    # 最大30秒待つ
    await asyncio.sleep(0.1)
    logger.info("Shutdown complete")
```

#### 2. ヘルスチェックエンドポイントの活用

```python
# 新しいエンドポイント
@app.get("/api/v1/ready")
async def readiness_check():
    """新しいリクエストを受け付ける準備ができているか"""
    if shutdown_event.is_set():
        raise HTTPException(status_code=503, detail="Server is shutting down")
    return {"status": "ready"}
```

#### 3. ブルーグリーンデプロイメント（簡易版）

```bash
#!/bin/bash
# scripts/safe_deploy.sh

# 新しいコンテナを別名で起動
docker run -d \
    --name logo-detection-api-new \
    -p 8001:8000 \
    $DOCKER_IMAGE

# ヘルスチェック
for i in {1..30}; do
    if curl -f http://localhost:8001/api/v1/health; then
        echo "New container is healthy"
        
        # 古いコンテナにSIGTERMを送信（グレースフルシャットダウン）
        docker kill --signal=SIGTERM logo-detection-api
        
        # 30秒待つ（処理中のリクエスト完了待ち）
        sleep 30
        
        # 古いコンテナを削除
        docker rm logo-detection-api
        
        # 新しいコンテナをリネーム
        docker rename logo-detection-api-new logo-detection-api
        
        # ポートを切り替え（iptablesやnginxで）
        break
    fi
    sleep 2
done
```

## 実装の推奨事項

### 1. 非同期デプロイ

```python
@app.route("/git/pull", methods=["POST"])
async def git_pull():
    # バックグラウンドでデプロイを実行
    deploy_task_id = str(uuid.uuid4())
    
    # タスクをキューに追加
    deployment_queue.put({
        "id": deploy_task_id,
        "rebuild": request.json.get("rebuild", False)
    })
    
    return jsonify({
        "task_id": deploy_task_id,
        "message": "Deployment task queued",
        "status_url": f"/deployment/status/{deploy_task_id}"
    }), 202  # Accepted
```

### 2. デプロイメント前チェック

```python
def check_if_rebuild_needed(repo_dir):
    """コミット間の差分を確認して再ビルドが必要か判定"""
    
    # 最後にビルドしたコミットハッシュを記録
    last_build_file = "/opt/logo-detection/.last_build_commit"
    
    # 現在のコミットハッシュ
    current_commit = run_command(f"cd {repo_dir} && git rev-parse HEAD")["stdout"].strip()
    
    # 最後のビルド時のコミット
    if os.path.exists(last_build_file):
        with open(last_build_file, "r") as f:
            last_commit = f.read().strip()
    else:
        return True  # 初回は必ずビルド
    
    # 差分をチェック
    diff_result = run_command(
        f"cd {repo_dir} && git diff --name-only {last_commit} {current_commit}"
    )
    
    changed_files = diff_result["stdout"].strip().split("\n")
    
    # 再ビルドが必要なファイルパターン
    rebuild_patterns = [
        "requirements.txt",
        "Dockerfile",
        "src/",
        ".dockerignore"
    ]
    
    for file in changed_files:
        for pattern in rebuild_patterns:
            if pattern in file:
                return True
    
    return False
```

## 実運用での推奨フロー

### 開発環境
1. コード変更をプッシュ
2. 自動で再ビルドが必要か判定
3. 必要なければ設定の再読み込みのみ

### 本番環境
1. ステージング環境でテスト
2. カナリアデプロイメント（一部のトラフィックのみ新バージョンへ）
3. 問題なければ全体切り替え

### 緊急時のロールバック
```bash
# 前のイメージにタグをつけておく
docker tag $DOCKER_IMAGE $DOCKER_IMAGE:rollback

# ロールバック実行
docker stop logo-detection-api
docker rm logo-detection-api
docker run -d --name logo-detection-api $DOCKER_IMAGE:rollback
```