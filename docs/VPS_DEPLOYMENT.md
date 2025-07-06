# VPS Deployment Guide

## 推奨スペック

- **最小要件**: 2 vCPU, 2GB RAM, 20GB SSD
- **推奨要件**: 4 vCPU, 4GB RAM, 40GB SSD
- **OS**: Ubuntu 22.04 LTS

## クイックデプロイ

新しいVPSで以下のコマンドを実行するだけでデプロイが完了します：

```bash
wget https://raw.githubusercontent.com/kentatsujikawadev/logo-detection-api/main/scripts/vps_setup.sh
chmod +x vps_setup.sh
sudo ./vps_setup.sh
```

## セットアップ内容

1. **Dockerのインストール**
2. **ディレクトリ構造の作成**
   ```
   /opt/logo-detection-api/
   ├── models/      # YOLOモデル
   ├── logs/        # ログファイル
   ├── data/        # データファイル
   └── scripts/     # 管理スクリプト
   ```

3. **管理APIサーバー** (ポート8080)
   - コンテナのステータス確認
   - Dockerイメージの更新と再起動
   - ログの確認
   - 設定の更新

4. **環境変数設定** (2コア2GB向け最適化)
   ```
   MAX_CONCURRENT_DETECTIONS=2
   MAX_CONCURRENT_DOWNLOADS=15
   MAX_BATCH_SIZE=30
   ```

## 使用方法

### APIエンドポイント

- **Logo Detection API**: `http://your-vps-ip:8000`
- **API Documentation**: `http://your-vps-ip:8000/docs`
- **Batch UI**: `http://your-vps-ip:8000/ui/batch`

### 管理コマンド

#### コンテナの状態確認
```bash
curl http://localhost:8080/
```

#### 最新版に更新
```bash
curl -X POST http://localhost:8080/pull-restart
```

#### ログ確認
```bash
# 最新100行
curl http://localhost:8080/logs

# 最新500行
curl http://localhost:8080/logs?lines=500
```

#### 設定確認
```bash
curl http://localhost:8080/config
```

#### 設定更新
```bash
curl -X POST http://localhost:8080/config \
  -H 'Content-Type: application/json' \
  -d '{
    "MAX_CONCURRENT_DETECTIONS": "3",
    "MAX_CONCURRENT_DOWNLOADS": "20"
  }'
```

設定更新後はコンテナの再起動が必要です：
```bash
curl -X POST http://localhost:8080/pull-restart
```

## パフォーマンスチューニング

### 2コア2GB VPS向け
```json
{
  "MAX_CONCURRENT_DETECTIONS": "2",
  "MAX_CONCURRENT_DOWNLOADS": "15",
  "MAX_BATCH_SIZE": "30"
}
```

### 4コア4GB VPS向け
```json
{
  "MAX_CONCURRENT_DETECTIONS": "4",
  "MAX_CONCURRENT_DOWNLOADS": "30",
  "MAX_BATCH_SIZE": "50"
}
```

### 8コア8GB VPS向け
```json
{
  "MAX_CONCURRENT_DETECTIONS": "6",
  "MAX_CONCURRENT_DOWNLOADS": "50",
  "MAX_BATCH_SIZE": "100"
}
```

## トラブルシューティング

### コンテナが起動しない場合
```bash
# コンテナのログを確認
docker logs logo-detection-api

# 管理サーバーのログを確認
sudo journalctl -u logo-detection-manager -f
```

### メモリ不足の場合
1. 設定を調整
   ```bash
   curl -X POST http://localhost:8080/config \
     -H 'Content-Type: application/json' \
     -d '{"MAX_BATCH_SIZE": "20", "MAX_CONCURRENT_DETECTIONS": "1"}'
   ```

2. コンテナを再起動
   ```bash
   curl -X POST http://localhost:8080/pull-restart
   ```

### ポートが使用中の場合
```bash
# 使用中のポートを確認
sudo lsof -i :8000
sudo lsof -i :8080

# 必要に応じてプロセスを停止
sudo kill -9 <PID>
```

## セキュリティ設定

### ファイアウォール設定
```bash
# UFWを使用する場合
sudo ufw allow 8000/tcp  # API
sudo ufw allow 8080/tcp  # 管理API (必要に応じて制限)
sudo ufw enable
```

### 管理APIへのアクセス制限
本番環境では管理API（8080ポート）へのアクセスを制限することを推奨：

```bash
# 特定のIPからのみアクセスを許可
sudo ufw allow from YOUR_IP to any port 8080
```

## バックアップ

### モデルとデータのバックアップ
```bash
# バックアップスクリプト
cat << 'EOF' > /opt/logo-detection-api/scripts/backup.sh
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/backups/logo-detection-api"
mkdir -p $BACKUP_DIR

# モデル、ログ、データをバックアップ
tar -czf $BACKUP_DIR/backup_$DATE.tar.gz \
  -C /opt/logo-detection-api \
  models logs data .env

# 古いバックアップを削除（7日以上）
find $BACKUP_DIR -name "backup_*.tar.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_DIR/backup_$DATE.tar.gz"
EOF

chmod +x /opt/logo-detection-api/scripts/backup.sh

# Cronジョブに追加（毎日午前3時に実行）
(crontab -l 2>/dev/null; echo "0 3 * * * /opt/logo-detection-api/scripts/backup.sh") | crontab -
```

## モニタリング

### 簡易モニタリングスクリプト
```bash
cat << 'EOF' > /opt/logo-detection-api/scripts/monitor.sh
#!/bin/bash
while true; do
  clear
  echo "=== Logo Detection API Monitor ==="
  echo "Time: $(date)"
  echo ""
  
  # コンテナステータス
  echo "Container Status:"
  docker ps --filter name=logo-detection-api --format "table {{.Status}}\t{{.Ports}}"
  echo ""
  
  # リソース使用状況
  echo "Resource Usage:"
  docker stats --no-stream logo-detection-api
  echo ""
  
  # 最新のログ
  echo "Recent Logs:"
  docker logs --tail 10 logo-detection-api
  
  sleep 5
done
EOF

chmod +x /opt/logo-detection-api/scripts/monitor.sh
```

実行：
```bash
/opt/logo-detection-api/scripts/monitor.sh
```