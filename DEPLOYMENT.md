# Logo Detection API - VPS展開ガイド

## 概要

このガイドでは、Logo Detection APIを新しいVPSに展開する手順を説明します。

## 前提条件

- Ubuntu 22.04 LTS を搭載したVPS
- root権限またはsudo権限を持つユーザー
- 最低2GB RAM（推奨: 4GB以上）
- 20GB以上のディスク容量
- ポート80, 443, 8000へのアクセス許可

## クイックスタート

新しいVPSで以下のコマンドを実行するだけで展開できます：

```bash
# リポジトリURLを環境に合わせて変更してください
export REPO_URL="https://github.com/YOUR_USERNAME/logo-detection-api.git"

# セットアップスクリプトをダウンロードして実行
curl -sSL https://raw.githubusercontent.com/YOUR_USERNAME/logo-detection-api/main/setup.sh | sudo bash
```

## 詳細手順

### 1. VPSの初期設定

```bash
# システムを最新に更新
sudo apt update && sudo apt upgrade -y

# タイムゾーンを設定（日本の場合）
sudo timedatectl set-timezone Asia/Tokyo
```

### 2. セットアップスクリプトの実行

```bash
# プロジェクトをクローン
git clone https://github.com/YOUR_USERNAME/logo-detection-api.git
cd logo-detection-api

# セットアップスクリプトに実行権限を付与
chmod +x setup.sh

# セットアップを実行
sudo ./setup.sh
```

### 3. 環境設定

```bash
# .envファイルを編集
cd /opt/logo-detection-api
sudo vim .env
```

以下の設定を確認・変更：

```env
# API設定
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=production

# モデル設定
MODEL_PATH=models/yolov8n.pt
CONFIDENCE_THRESHOLD=0.7

# セキュリティ設定
UPDATE_WEBHOOK_SECRET=your-secure-webhook-secret
ADMIN_TOKEN=your-secure-admin-token
```

### 4. モデルファイルの配置

```bash
# モデルディレクトリに移動
cd /opt/logo-detection-api/models

# 既存のモデルをコピー（例）
sudo cp /path/to/your/model.pt ./

# または、Google Colabで学習したモデルをダウンロード
sudo wget -O colab_model.pt "https://your-storage.com/model.pt"

# 権限を設定
sudo chown -R 1000:1000 /opt/logo-detection-api/models
```

### 5. サービスの起動

```bash
cd /opt/logo-detection-api

# Dockerイメージをビルド
sudo docker-compose -f docker-compose.production.yml build

# サービスを起動
sudo docker-compose -f docker-compose.production.yml up -d

# ログを確認
sudo docker-compose -f docker-compose.production.yml logs -f
```

### 6. 動作確認

```bash
# ヘルスチェック
curl http://localhost:8000/health

# バージョン情報
curl http://localhost:8000/api/v1/system/version

# APIドキュメント
# ブラウザで http://your-vps-ip/docs にアクセス
```

## 自動更新の設定

### Webhook経由での更新

GitHubのWebhookを設定して、pushイベントで自動更新：

1. GitHubリポジトリの Settings → Webhooks → Add webhook
2. Payload URL: `http://your-vps-ip/api/v1/system/update`
3. Content type: `application/json`
4. Secret: `.env`の`UPDATE_WEBHOOK_SECRET`と同じ値
5. Events: `Push events`を選択

### 手動更新

```bash
# 更新スクリプトを実行
sudo /opt/logo-detection-api/update.sh

# または、APIエンドポイント経由
curl -X POST http://localhost:8000/api/v1/system/update \
  -H "X-Admin-Token: your-admin-token"
```

## SSL証明書の設定（推奨）

Let's Encryptを使用した無料SSL証明書の設定：

```bash
# Certbotをインストール
sudo apt install certbot python3-certbot-nginx -y

# 証明書を取得（your-domain.comを実際のドメインに変更）
sudo certbot --nginx -d your-domain.com

# 自動更新の確認
sudo certbot renew --dry-run
```

## 監視とメンテナンス

### ログの確認

```bash
# アプリケーションログ
tail -f /opt/logo-detection-api/logs/app.log

# Dockerコンテナログ
sudo docker-compose -f docker-compose.production.yml logs -f api

# Nginxログ
tail -f /opt/logo-detection-api/logs/nginx/access.log
```

### リソース使用状況

```bash
# Dockerコンテナの状態
sudo docker ps
sudo docker stats

# システムリソース
htop
df -h
```

### バックアップ

```bash
# モデルファイルのバックアップ
sudo tar -czf models-backup-$(date +%Y%m%d).tar.gz /opt/logo-detection-api/models

# 設定ファイルのバックアップ
sudo tar -czf config-backup-$(date +%Y%m%d).tar.gz /opt/logo-detection-api/.env /opt/logo-detection-api/nginx
```

## トラブルシューティング

### APIが起動しない

```bash
# コンテナの状態を確認
sudo docker-compose -f docker-compose.production.yml ps

# エラーログを確認
sudo docker-compose -f docker-compose.production.yml logs api

# コンテナを再起動
sudo docker-compose -f docker-compose.production.yml restart api
```

### メモリ不足エラー

```bash
# スワップファイルを作成（2GB）
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 永続化
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### ポートが使用中

```bash
# 使用中のポートを確認
sudo lsof -i :8000
sudo lsof -i :80

# 必要に応じてプロセスを停止
sudo kill -9 <PID>
```

## セキュリティ推奨事項

1. **ファイアウォール設定**
   - 必要なポートのみ開放
   - SSH接続元IPを制限

2. **定期的な更新**
   - システムパッケージの更新
   - Dockerイメージの更新
   - セキュリティパッチの適用

3. **アクセス制限**
   - 管理エンドポイントへのIP制限
   - 強力なパスワード/トークンの使用
   - HTTPS通信の強制

4. **監視**
   - ログの定期的な確認
   - 異常なアクセスパターンの検知
   - リソース使用状況の監視

## 複数VPSでの運用

同じ設定で複数のVPSを運用する場合：

1. **設定の共通化**
   ```bash
   # 共通の.envファイルを作成
   # 各VPS固有の設定のみ環境変数で上書き
   ```

2. **ロードバランサーの設定**
   - Nginx/HAProxyでの負荷分散
   - ヘルスチェックの設定

3. **デプロイの自動化**
   - Ansible/Terraformの使用
   - CI/CDパイプラインの構築

## サポート

問題が発生した場合：

1. エラーログを確認
2. [Issues](https://github.com/YOUR_USERNAME/logo-detection-api/issues)で報告
3. ドキュメントの[FAQ](#)を参照