# Logo Detection API - Phase 2

高性能なロゴ検出APIシステム（商標ロゴ専用モデル対応版）。YOLOv8を使用して数百万枚の画像処理に対応し、複数モデル管理と日本語ブランド正規化機能を提供します。

## 概要

このAPIは、DB側からのPushリクエストを受信して画像バッチを並列処理するロゴ検出システムです。Phase 2では商標ロゴ専用モデルサポートとブランド分類機能を追加し、より高精度な商標ロゴ検出を実現します。

### Phase 2 新機能

- **🎯 複数モデル対応**: 汎用・商標専用・カスタムモデルの動的切り替え
- **🌏 多言語対応**: 日本語・英語ブランド名の自動正規化
- **📊 カテゴリ分類**: 玩具・電子機器・自動車等の業界別分類
- **🔄 動的モデル管理**: API経由でのモデル切り替え・読み込み
- **📈 信頼度調整**: カテゴリ別の適応的信頼度閾値

### 基本機能

- **バッチ処理**: 最大100枚の画像を並列処理
- **単一画像処理**: 個別画像の即座な処理
- **非同期処理**: aiohttp + asyncioによる高効率な並列ダウンロード
- **CPU最適化**: PyTorch CPUモードでの高速推論
- **監視機能**: リアルタイムメトリクス、ヘルスチェック
- **スケーラビリティ**: ステートレス設計による水平スケーリング対応
- **Docker対応**: 開発・本番環境での容易なデプロイ

## システム要件

- **OS**: WSL2 Ubuntu 22.04 以上
- **Python**: 3.10 以上
- **メモリ**: 4GB 以上推奨
- **CPU**: 4コア以上推奨
- **ディスク**: 2GB 以上の空き容量

## 環境構築

### 1. WSL2 Ubuntu環境での構築

```bash
# 1. リポジトリのクローン
git clone <repository-url>
cd logo-detection-api

# 2. Python仮想環境の作成
python3 -m venv venv
source venv/bin/activate

# 3. 依存関係のインストール
pip install --upgrade pip
pip install -r requirements.txt

# 4. PyTorch CPUバージョンのインストール
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. 環境変数の設定
cp .env.example .env
# .envファイルを環境に合わせて編集

# 6. モデルディレクトリの作成
mkdir -p models logs temp

# 7. アプリケーションの起動
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Docker環境での構築

#### 開発環境

```bash
# 開発環境での起動（ホットリロード有効）
docker-compose --profile dev up --build

# またはdocker runで直接起動
docker build -f docker/Dockerfile.dev -t logo-detection-api:dev .
docker run -p 8000:8000 -v $(pwd):/app logo-detection-api:dev
```

#### 本番環境

```bash
# 本番環境での起動
docker-compose --profile prod up --build -d

# 複数インスタンスでの起動
docker-compose --profile prod up --scale logo-detection-api-prod=3 -d
```

### 3. VPS展開（クイックスタート）

新しいVPS（Ubuntu 22.04 LTS）で以下のコマンドを実行するだけで展開可能：

```bash
# セットアップスクリプトをダウンロードして実行
curl -sSL https://raw.githubusercontent.com/YOUR_USERNAME/logo-detection-api/main/setup.sh | sudo bash
```

詳細な展開手順は[DEPLOYMENT.md](DEPLOYMENT.md)を参照してください。

## API仕様

### エンドポイント一覧

| メソッド | エンドポイント | 説明 |
|---------|---------------|------|
| GET | `/` | API情報 |
| GET | `/docs` | Swagger UI |
| GET | `/api/v1/health` | ヘルスチェック |
| GET | `/api/v1/metrics` | メトリクス取得 |
| POST | `/api/v1/process/batch` | バッチ処理 |
| POST | `/api/v1/process/single` | 単一画像処理 |
| **GET** | **`/api/v1/models`** | **利用可能モデル一覧** |
| **POST** | **`/api/v1/models/switch`** | **モデル切り替え** |
| **GET** | **`/api/v1/models/current`** | **現在のモデル情報** |
| **GET** | **`/api/v1/brands`** | **登録ブランド一覧** |
| **GET** | **`/api/v1/categories`** | **ブランドカテゴリ一覧** |
| **GET** | **`/api/v1/brands/{brand}/info`** | **ブランド詳細情報** |
| **POST** | **`/api/v1/system/update`** | **システム更新（Git pull）** |
| **GET** | **`/api/v1/system/version`** | **バージョン情報** |
| **POST** | **`/api/v1/system/restart`** | **システム再起動** |

### 1. バッチ処理 API

**リクエスト例:**

```bash
curl -X POST "http://localhost:8000/api/v1/process/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "batch_id": "batch_20241226_001",
    "images": [
      {
        "image_id": "img_001",
        "image_url": "https://example.com/image1.jpg"
      },
      {
        "image_id": "img_002",
        "image_url": "https://example.com/image2.jpg"
      }
    ],
    "options": {
      "confidence_threshold": 0.8,
      "max_detections": 10
    }
  }'
```

**レスポンス例:**

```json
{
  "batch_id": "batch_20241226_001",
  "processing_time": 2.34,
  "total_images": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    {
      "image_id": "img_001",
      "detections": [
        {
          "logo_name": "BANDAI",
          "confidence": 0.95,
          "bbox": [100, 50, 200, 100]
        }
      ],
      "processing_time": 0.045,
      "status": "success"
    }
  ],
  "errors": []
}
```

### 2. 単一画像処理 API

**リクエスト例:**

```bash
curl -X POST "http://localhost:8000/api/v1/process/single" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "confidence_threshold": 0.8,
    "max_detections": 10
  }'
```

**レスポンス例:**

```json
{
  "detections": [
    {
      "logo_name": "BANDAI",
      "confidence": 0.95,
      "bbox": [100, 50, 200, 100]
    }
  ],
  "processing_time": 0.045,
  "status": "success"
}
```

### 3. ヘルスチェック API

```bash
curl "http://localhost:8000/api/v1/health"
```

**レスポンス例:**

```json
{
  "status": "ok",
  "timestamp": "2024-12-26T10:30:00Z",
  "model_loaded": true,
  "system_info": {
    "cpu_count": 4,
    "memory_gb": 8.0,
    "memory_used_percent": 45.2
  }
}
```

### 4. モデル管理 API（新機能）

**利用可能モデル一覧:**

```bash
curl "http://localhost:8000/api/v1/models"
```

**レスポンス例:**

```json
{
  "current_model": "general",
  "models": {
    "general": {
      "name": "general",
      "path": "models/yolov8n.pt",
      "loaded": true,
      "is_current": true,
      "confidence_threshold": 0.8
    },
    "trademark": {
      "name": "trademark", 
      "path": "models/trademark_logos.pt",
      "loaded": false,
      "is_current": false,
      "confidence_threshold": 0.7
    }
  },
  "total_models": 2,
  "loaded_models": 1
}
```

**モデル切り替え:**

```bash
curl -X POST "http://localhost:8000/api/v1/models/switch?model_name=trademark"
```

**レスポンス例:**

```json
{
  "success": true,
  "message": "Successfully switched to model 'trademark'",
  "new_model": "trademark",
  "model_info": {
    "loaded": true,
    "current_model": "trademark",
    "brand_classification_enabled": true
  }
}
```

### 5. ブランド管理 API（新機能）

**登録ブランド一覧:**

```bash
curl "http://localhost:8000/api/v1/brands"
```

**レスポンス例:**

```json
[
  {
    "key": "BANDAI",
    "japanese": "バンダイ",
    "english": "BANDAI", 
    "official_name": "株式会社バンダイ",
    "category": "玩具・ゲーム",
    "category_en": "Toys & Games"
  },
  {
    "key": "NINTENDO",
    "japanese": "任天堂",
    "english": "NINTENDO",
    "official_name": "任天堂株式会社", 
    "category": "玩具・ゲーム",
    "category_en": "Toys & Games"
  }
]
```

**ブランド詳細情報:**

```bash
curl "http://localhost:8000/api/v1/brands/BANDAI/info"
```

**レスポンス例:**

```json
{
  "brand_info": {
    "original": "BANDAI",
    "normalized": "BANDAI", 
    "japanese": "バンダイ",
    "english": "BANDAI",
    "official_name": "株式会社バンダイ",
    "aliases": ["BANDAI", "バンダイ", "ばんだい"]
  },
  "category_info": {
    "category": {
      "key": "toys_games",
      "name": "玩具・ゲーム",
      "name_en": "Toys & Games"
    },
    "subcategory": {
      "key": "toys", 
      "name": "玩具",
      "name_en": "Toys"
    }
  },
  "confidence_adjustment": -0.05
}
```

### 6. 拡張された検出結果（新機能）

商標専用モデル使用時は、検出結果にブランド正規化情報が追加されます：

```json
{
  "detections": [
    {
      "logo_name": "バンダイ",
      "confidence": 0.95,
      "bbox": [100, 50, 200, 100],
      "brand_info": {
        "original": "BANDAI",
        "normalized": "BANDAI",
        "japanese": "バンダイ", 
        "english": "BANDAI",
        "official_name": "株式会社バンダイ",
        "aliases": ["BANDAI", "バンダイ", "ばんだい"]
      },
      "category_info": {
        "category": {
          "key": "toys_games",
          "name": "玩具・ゲーム",
          "name_en": "Toys & Games"
        },
        "subcategory": {
          "key": "toys",
          "name": "玩具", 
          "name_en": "Toys"
        }
      },
      "model_used": "trademark",
      "original_confidence": 0.90,
      "raw_detection": "BANDAI"
    }
  ],
  "processing_time": 0.045,
  "status": "success"
}
```

## 設定オプション

### 環境変数（.env ファイル）

```bash
# API設定
HOST=0.0.0.0
PORT=8000
DEBUG=false

# モデル設定
MODEL_PATH=models/yolov8n.pt
CONFIDENCE_THRESHOLD=0.8
MAX_DETECTIONS=10

# バッチ処理設定
MAX_BATCH_SIZE=100
MAX_CONCURRENT_DOWNLOADS=50
MAX_CONCURRENT_DETECTIONS=10
DOWNLOAD_TIMEOUT=30
PROCESSING_TIMEOUT=300

# 画像処理設定
MAX_IMAGE_SIZE=1920
SUPPORTED_FORMATS=["jpg", "jpeg", "png", "bmp", "webp"]

# ログ設定
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# パフォーマンス監視
ENABLE_METRICS=true
METRICS_RETENTION_HOURS=24
```

## テスト

### テスト実行

```bash
# 全テスト実行
pytest

# 詳細出力でテスト実行
pytest -v

# カバレッジ付きテスト実行
pytest --cov=src tests/

# 特定のテストファイル実行
pytest tests/test_api.py
pytest tests/test_batch_processing.py
```

### テスト用スクリプト例

#### 1. バッチ処理テスト

```python
import asyncio
import aiohttp

async def test_batch_processing():
    """バッチ処理のテスト"""
    batch_data = {
        "batch_id": "test_batch_001",
        "images": [
            {
                "image_id": "test_img_001",
                "image_url": "https://via.placeholder.com/640x480.jpg"
            },
            {
                "image_id": "test_img_002", 
                "image_url": "https://via.placeholder.com/800x600.jpg"
            }
        ],
        "options": {
            "confidence_threshold": 0.8,
            "max_detections": 10
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/api/v1/process/batch",
            json=batch_data
        ) as response:
            result = await response.json()
            print(f"Batch processed: {result['successful']}/{result['total_images']}")
            print(f"Processing time: {result['processing_time']:.2f}s")

# 実行
asyncio.run(test_batch_processing())
```

#### 2. 単一画像処理テスト

```python
import requests

def test_single_image():
    """単一画像処理のテスト"""
    data = {
        "image_url": "https://via.placeholder.com/640x480.jpg",
        "confidence_threshold": 0.7,
        "max_detections": 5
    }
    
    response = requests.post(
        "http://localhost:8000/api/v1/process/single",
        json=data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Detections found: {len(result['detections'])}")
        print(f"Processing time: {result['processing_time']:.3f}s")
        for detection in result['detections']:
            print(f"- {detection['logo_name']}: {detection['confidence']:.2f}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# 実行
test_single_image()
```

## パフォーマンスチューニング

### 1. 並列処理の最適化

```bash
# CPU集約的な処理に適した設定例
MAX_CONCURRENT_DOWNLOADS=50
MAX_CONCURRENT_DETECTIONS=4  # CPU コア数に応じて調整

# メモリ使用量を重視する場合
MAX_CONCURRENT_DOWNLOADS=20
MAX_CONCURRENT_DETECTIONS=2
```

### 2. モデル最適化

```python
# カスタムモデルの使用例
MODEL_PATH=models/custom_logo_model.pt

# 信頼度閾値の調整による高速化
CONFIDENCE_THRESHOLD=0.9  # 高い閾値で候補を絞る
```

### 3. システムレベル最適化

```bash
# PyTorchのスレッド数制限
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# メモリアロケータの最適化
export MALLOC_TRIM_THRESHOLD_=100000
```

## 監視とメトリクス

### メトリクス取得

```bash
curl "http://localhost:8000/api/v1/metrics"
```

### 主要メトリクス

- `total_processed`: 処理済み画像総数
- `avg_processing_time`: 平均処理時間
- `error_rate`: エラー率
- `uptime_seconds`: 稼働時間
- `active_batches`: アクティブバッチ数

### ログ監視

```bash
# リアルタイムログ監視
tail -f logs/app.log

# エラーログの抽出
grep ERROR logs/app.log

# 処理時間の統計
grep "processing_time" logs/app.log | awk '{print $NF}' | sort -n
```

## トラブルシューティング

### よくある問題と解決方法

#### 1. モデル読み込みエラー

```bash
# エラー: Model file not found
mkdir -p models
# 初回起動時に自動的にYOLOv8nモデルがダウンロードされます
```

#### 2. メモリ不足エラー

```bash
# 並列数を減らす
MAX_CONCURRENT_DOWNLOADS=10
MAX_CONCURRENT_DETECTIONS=2

# 画像サイズを制限
MAX_IMAGE_SIZE=1280
```

#### 3. WSL2でのポートアクセス問題

```bash
# Windowsファイアウォールの設定確認
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=WSL2_IP

# WSL2のIPアドレス確認
wsl hostname -I
```

#### 4. Docker環境での問題

```bash
# イメージの再ビルド
docker-compose down
docker-compose build --no-cache
docker-compose up

# ボリュームのクリア
docker volume prune
```

### ログレベルの調整

```bash
# 開発時: 詳細ログ
LOG_LEVEL=DEBUG

# 本番時: 必要最小限のログ
LOG_LEVEL=WARNING
```

## 大規模運用時の注意点

### 1. 水平スケーリング

```yaml
# docker-compose.yml での複数インスタンス起動
services:
  logo-detection-api:
    scale: 3  # 3インスタンス起動
```

### 2. ロードバランサ設定

```nginx
# nginx.conf の例
upstream logo_detection_backend {
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}

server {
    listen 80;
    location / {
        proxy_pass http://logo_detection_backend;
    }
}
```

### 3. モニタリング

```bash
# Prometheusメトリクス出力（将来実装予定）
curl "http://localhost:8000/metrics/prometheus"

# ヘルスチェック用エンドポイント
curl "http://localhost:8000/api/v1/readiness"  # Kubernetes readiness probe
curl "http://localhost:8000/api/v1/liveness"   # Kubernetes liveness probe
```

### 4. セキュリティ対策

```bash
# 本番環境での推奨設定
CORS_ORIGINS=["https://your-domain.com"]
MAX_REQUEST_SIZE=50MB
DEBUG=false
```

## API仕様詳細

詳細なAPI仕様については、サーバー起動後に以下のURLでSwagger UIを確認してください：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 新しい企業ロゴ追加手順

### 手順概要

新しい企業ロゴを追加して学習させる方法は2つあります：

1. **自動データセット生成を使用する方法**（推奨）
2. **手動でアノテーションする方法**

### 方法1: 自動データセット生成を使用する方法

#### 1. ロゴ画像の準備

```bash
# ロゴ画像を以下のディレクトリに配置
data/logos/input/
├── 新企業名A/
│   ├── logo1.jpg
│   ├── logo2.png
│   └── logo3.webp
└── 新企業名B/
    ├── logo1.jpg
    └── logo2.png
```

**注意点:**
- ディレクトリ名 = 企業名（クラス名）
- 対応形式: JPG, PNG, WEBP, BMP
- 1企業につき最低1枚、推奨は3-5枚
- 画像サイズ: 最大1920px（自動リサイズされます）

#### 2. サーバー起動

```bash
# 仮想環境の有効化
source venv/bin/activate

# サーバー起動
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 3. 自動データセット生成

```bash
# 新しい企業ロゴ用のデータセットを自動生成
curl -X POST "http://localhost:8000/api/v1/training/datasets/new_company_logos/generate-sample" \
  -H "Content-Type: application/json" \
  -d '{
    "classes": ["新企業名A", "新企業名B"],
    "images_per_class": 30
  }'
```

**パラメータ説明:**
- `classes`: 追加したい企業名のリスト
- `images_per_class`: 各企業につき生成する画像数（5-100枚）

#### 4. データセット検証

```bash
# データセットの統計情報を確認
curl "http://localhost:8000/api/v1/training/datasets/new_company_logos/stats"

# データセットの有効性を検証
curl "http://localhost:8000/api/v1/training/datasets/new_company_logos/validate"
```

#### 5. 学習実行

```bash
# 学習開始
curl -X POST "http://localhost:8000/api/v1/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "company_logos_v2",
    "dataset_name": "new_company_logos",
    "base_model": "yolov8n.pt",
    "epochs": 50,
    "batch_size": 16,
    "learning_rate": 0.001
  }'
```

#### 6. 学習進捗確認

```bash
# 学習状況確認
curl "http://localhost:8000/api/v1/training/status"

# 詳細進捗確認
curl "http://localhost:8000/api/v1/training/progress"
```

### 方法2: 手動でアノテーションする方法

#### 1. ロゴ画像の準備

```bash
# 高品質なロゴ画像を配置
data/logos/input/
├── 新企業名A/
│   ├── logo_clear.jpg      # 背景がクリアなロゴ
│   ├── logo_document.jpg   # 文書内のロゴ
│   └── logo_product.jpg    # 製品上のロゴ
└── 新企業名B/
    ├── logo_sign.jpg       # 看板のロゴ
    └── logo_package.jpg    # パッケージのロゴ
```

#### 2. データセット作成

```bash
# 空のデータセットを作成
curl -X POST "http://localhost:8000/api/v1/training/datasets/create" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "manual_company_logos",
    "classes": ["新企業名A", "新企業名B"],
    "description": "手動アノテーション企業ロゴデータセット"
  }'
```

#### 3. 画像とアノテーション追加

各画像に対してバウンディングボックスを手動で指定：

```bash
# 画像1のアノテーション追加
curl -X POST "http://localhost:8000/api/v1/training/datasets/manual_company_logos/add-image" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "data/logos/input/新企業名A/logo_clear.jpg",
    "annotations": [
      {
        "class_name": "新企業名A",
        "bbox": [100, 50, 300, 150],
        "confidence": 1.0
      }
    ],
    "split": "train"
  }'

# 画像2のアノテーション追加
curl -X POST "http://localhost:8000/api/v1/training/datasets/manual_company_logos/add-image" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "data/logos/input/新企業名A/logo_document.jpg",
    "annotations": [
      {
        "class_name": "新企業名A",
        "bbox": [250, 100, 450, 200],
        "confidence": 1.0
      }
    ],
    "split": "train"
  }'

# 検証用画像の追加
curl -X POST "http://localhost:8000/api/v1/training/datasets/manual_company_logos/add-image" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "data/logos/input/新企業名A/logo_product.jpg",
    "annotations": [
      {
        "class_name": "新企業名A",
        "bbox": [150, 200, 350, 300],
        "confidence": 1.0
      }
    ],
    "split": "val"
  }'
```

**バウンディングボックスの指定方法:**
- `bbox`: [x_min, y_min, x_max, y_max]
- 座標は画像の左上を(0,0)とするピクセル座標
- x_min, y_min: ロゴの左上角
- x_max, y_max: ロゴの右下角

#### 4. データセット分割（オプション）

```bash
# 学習・検証・テストセットに自動分割
curl -X POST "http://localhost:8000/api/v1/training/datasets/manual_company_logos/split" \
  -H "Content-Type: application/json" \
  -d '{
    "train_ratio": 0.7,
    "val_ratio": 0.2,
    "test_ratio": 0.1
  }'
```

#### 5. データセット検証

```bash
# 統計情報確認
curl "http://localhost:8000/api/v1/training/datasets/manual_company_logos/stats"

# 検証実行
curl "http://localhost:8000/api/v1/training/datasets/manual_company_logos/validate"
```

#### 6. 学習実行

```bash
# 学習開始
curl -X POST "http://localhost:8000/api/v1/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "manual_company_logos_v1",
    "dataset_name": "manual_company_logos",
    "base_model": "yolov8n.pt",
    "epochs": 100,
    "batch_size": 8,
    "learning_rate": 0.0001
  }'
```

### 学習結果の確認と利用

#### 1. 学習完了確認

```bash
# 学習状況の確認
curl "http://localhost:8000/api/v1/training/status"

# 学習済みモデル一覧
curl "http://localhost:8000/api/v1/training/models"
```

#### 2. モデルの切り替え

```bash
# 新しく学習したモデルに切り替え
curl -X POST "http://localhost:8000/api/v1/models/switch?model_name=company_logos_v2"
```

#### 3. 検出テスト

```bash
# 単一画像での検出テスト
curl -X POST "http://localhost:8000/api/v1/process/single" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/test_logo_image.jpg",
    "confidence_threshold": 0.7,
    "max_detections": 10
  }'
```

### トラブルシューティング

#### よくある問題

1. **画像が見つからない**
   ```bash
   # 画像パスの確認
   ls -la data/logos/input/企業名/
   ```

2. **アノテーションエラー**
   ```bash
   # バウンディングボックスの座標が画像サイズを超えていないか確認
   # 座標は [x_min, y_min, x_max, y_max] の順序で指定
   ```

3. **学習が始まらない**
   ```bash
   # データセットの検証結果を確認
   curl "http://localhost:8000/api/v1/training/datasets/データセット名/validate"
   ```

4. **メモリ不足**
   ```bash
   # バッチサイズを小さくする
   "batch_size": 4
   ```

### 最適化のヒント

1. **自動データセット生成の場合:**
   - 元画像は高品質なものを使用
   - 企業ロゴが明確に識別できる画像を選択
   - 1企業につき3-5枚の異なる画像を用意

2. **手動アノテーションの場合:**
   - バウンディングボックスはロゴ部分を正確に囲む
   - 学習用に最低20枚、検証用に5枚程度を用意
   - 異なるサイズ、角度、背景の画像を含める

3. **学習パラメータ:**
   - epochs: 50-100（データ量に応じて調整）
   - batch_size: GPU使用時は16-32、CPU使用時は4-8
   - learning_rate: 0.001から開始、過学習の場合は0.0001に下げる

## 既存モデルへの追加学習（転移学習）

既存の学習済みモデルに新しい画像を追加して再学習する手順です。これにより、既存の検出性能を維持しながら新しいロゴを追加できます。

### 方法1: 既存データセットに画像を追加する方法

#### 1. 既存データセットの確認

```bash
# 利用可能なデータセット一覧を確認
curl "http://localhost:8000/api/v1/training/datasets"

# 特定のデータセットの詳細確認
curl "http://localhost:8000/api/v1/training/datasets/trademark_logos_final/stats"
```

#### 2. 新しいロゴ画像の準備

```bash
# 既存データセットに追加する画像を配置
data/logos/input/
├── 新企業名C/           # 新しく追加したいロゴ
│   ├── logo1.jpg
│   ├── logo2.png
│   └── logo3.webp
└── 既存企業名A/         # 既存企業の追加画像
    ├── new_logo1.jpg   # 新しい角度・背景のロゴ
    └── new_logo2.png   # 異なるサイズのロゴ
```

#### 3. 既存データセットにクラス追加（新企業の場合）

```bash
# 新しい企業クラスは add-image API で自動的に追加されます
# 既存データセットに新しいクラスの画像を追加する場合は下記の手動追加を使用
```

#### 4. 新しい画像をデータセットに追加

**自動追加の場合:**
```bash
# 新しい企業の画像を自動で生成（実際のAPI）
curl -X POST "http://localhost:8000/api/v1/training/datasets/new_company_addition/generate-sample" \
  -H "Content-Type: application/json" \
  -d '{
    "classes": ["新企業名C"],
    "images_per_class": 25
  }'
```

**手動追加の場合:**
```bash
# 新企業の画像を手動でアノテーション付きで追加
curl -X POST "http://localhost:8000/api/v1/training/datasets/trademark_logos_final/add-image" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "data/logos/input/新企業名C/logo1.jpg",
    "annotations": [
      {
        "class_name": "新企業名C",
        "bbox": [120, 80, 320, 180],
        "confidence": 1.0
      }
    ],
    "split": "train"
  }'

# 既存企業の追加画像も同様に追加
curl -X POST "http://localhost:8000/api/v1/training/datasets/trademark_logos_final/add-image" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "data/logos/input/既存企業名A/new_logo1.jpg",
    "annotations": [
      {
        "class_name": "既存企業名A",
        "bbox": [50, 30, 250, 130],
        "confidence": 1.0
      }
    ],
    "split": "train"
  }'
```

#### 5. データセット再検証

```bash
# 更新されたデータセットの統計確認
curl "http://localhost:8000/api/v1/training/datasets/trademark_logos_final/stats"

# データセットの有効性再検証
curl "http://localhost:8000/api/v1/training/datasets/trademark_logos_final/validate"
```

#### 6. 転移学習実行

```bash
# 既存の学習済みモデルをベースにした転移学習
curl -X POST "http://localhost:8000/api/v1/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "trademark_logos_v3",
    "dataset_name": "trademark_logos_final",
    "base_model": "models/trained/trademark_logos/weights/best.pt",
    "epochs": 30,
    "batch_size": 16,
    "learning_rate": 0.0001,
    "transfer_learning": true,
    "freeze_layers": 10
  }'
```

**転移学習パラメータの説明:**
- `base_model`: 既存の学習済みモデルのパス
- `transfer_learning`: 転移学習モードを有効化
- `freeze_layers`: 凍結する層数（既存の特徴を保持）
- `learning_rate`: 転移学習では小さめの学習率を使用
- `epochs`: 既存モデルがあるため少なめのエポック数

### 方法2: 新しいデータセットで増分学習

#### 1. 増分学習用データセット作成

```bash
# 新しい画像のみで増分学習用データセットを作成
curl -X POST "http://localhost:8000/api/v1/training/datasets/create" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "incremental_logos_v1",
    "classes": ["新企業名C", "新企業名D"],
    "description": "増分学習用データセット"
  }'
```

#### 2. 新しい画像の追加

```bash
# 新しい企業のロゴ画像を追加
curl -X POST "http://localhost:8000/api/v1/training/datasets/incremental_logos_v1/generate-sample" \
  -H "Content-Type: application/json" \
  -d '{
    "classes": ["新企業名C", "新企業名D"],
    "images_per_class": 20
  }'
```

#### 3. 増分学習実行

```bash
# 既存モデルをベースにした増分学習
curl -X POST "http://localhost:8000/api/v1/training/start-incremental" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "trademark_logos_incremental_v1",
    "new_dataset_name": "incremental_logos_v1",
    "base_model_path": "models/trained/trademark_logos/weights/best.pt",
    "original_classes": ["BANDAI", "Nintendo", "KONAMI", "Panasonic", "SONY"],
    "new_classes": ["新企業名C", "新企業名D"],
    "epochs": 25,
    "batch_size": 12,
    "learning_rate": 0.0001,
    "knowledge_distillation": true
  }'
```

**増分学習パラメータの説明:**
- `original_classes`: 元のモデルが学習済みのクラス
- `new_classes`: 新しく追加するクラス
- `knowledge_distillation`: 既存知識の蒸留を有効化
- より少ないエポック数で効率的に学習

### 方法3: モデル融合による統合

#### 1. 新しいロゴ専用モデルの学習

```bash
# 新しいロゴのみで専用モデルを学習
curl -X POST "http://localhost:8000/api/v1/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "new_logos_specialist",
    "dataset_name": "incremental_logos_v1",
    "base_model": "yolov8n.pt",
    "epochs": 50,
    "batch_size": 16,
    "learning_rate": 0.001
  }'
```

#### 2. モデル融合実行

```bash
# 既存モデルと新しいモデルを融合
curl -X POST "http://localhost:8000/api/v1/training/merge-models" \
  -H "Content-Type: application/json" \
  -d '{
    "output_model_name": "trademark_logos_merged_v1",
    "base_model_path": "models/trained/trademark_logos/weights/best.pt",
    "additional_model_path": "models/trained/new_logos_specialist/weights/best.pt",
    "merge_strategy": "weighted_average",
    "base_weight": 0.7,
    "additional_weight": 0.3
  }'
```

### 学習進捗の監視と評価

#### 1. 学習進捗確認

```bash
# リアルタイム学習状況
curl "http://localhost:8000/api/v1/training/status"

# 詳細進捗とメトリクス
curl "http://localhost:8000/api/v1/training/progress"

# 学習ログの確認
curl "http://localhost:8000/api/v1/training/logs?lines=50"
```

#### 2. モデル性能評価

```bash
# 検証データでの性能評価
curl -X POST "http://localhost:8000/api/v1/training/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "models/trained/trademark_logos_v3/weights/best.pt",
    "dataset_name": "trademark_logos_final",
    "split": "val"
  }'
```

#### 3. A/Bテスト用比較

```bash
# 旧モデルとの比較テスト
curl -X POST "http://localhost:8000/api/v1/training/compare-models" \
  -H "Content-Type: application/json" \
  -d '{
    "model_a": "models/trained/trademark_logos/weights/best.pt",
    "model_b": "models/trained/trademark_logos_v3/weights/best.pt",
    "test_dataset": "trademark_logos_final",
    "metrics": ["precision", "recall", "f1", "map50"]
  }'
```

### モデル管理と展開

#### 1. モデルのバックアップ

```bash
# 現在のモデルをバックアップ
curl -X POST "http://localhost:8000/api/v1/models/backup" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "trademark_logos_v2",
    "backup_name": "backup_before_v3_update"
  }'
```

#### 2. 段階的展開

```bash
# カナリアデプロイ（一部トラフィックで新モデルテスト）
curl -X POST "http://localhost:8000/api/v1/models/canary-deploy" \
  -H "Content-Type: application/json" \
  -d '{
    "new_model": "trademark_logos_v3",
    "traffic_percentage": 10,
    "duration_minutes": 60
  }'

# 問題なければ全体展開
curl -X POST "http://localhost:8000/api/v1/models/switch?model_name=trademark_logos_v3"
```

#### 3. ロールバック準備

```bash
# 問題発生時の即座ロールバック
curl -X POST "http://localhost:8000/api/v1/models/rollback" \
  -H "Content-Type: application/json" \
  -d '{
    "backup_name": "backup_before_v3_update"
  }'
```

### 転移学習のベストプラクティス

#### 1. 学習率の調整

```bash
# 段階的学習率調整
# Phase 1: 特徴抽出層のみ学習（低学習率）
"learning_rate": 0.0001
"freeze_layers": 15

# Phase 2: 全体ファインチューニング（更に低学習率）
"learning_rate": 0.00001  
"freeze_layers": 0
```

#### 2. データバランシング

```bash
# 既存クラスと新クラスのバランス確認
curl "http://localhost:8000/api/v1/training/datasets/trademark_logos_final/balance-report"

# 必要に応じてデータ拡張
curl -X POST "http://localhost:8000/api/v1/training/datasets/trademark_logos_final/augment" \
  -H "Content-Type: application/json" \
  -d '{
    "target_classes": ["新企業名C"],
    "target_samples_per_class": 30,
    "augmentation_types": ["rotation", "scaling", "brightness", "noise"]
  }'
```

#### 3. 性能劣化の防止

```bash
# 忘却防止のための定期的な全データ再学習
curl -X POST "http://localhost:8000/api/v1/training/schedule-retraining" \
  -H "Content-Type: application/json" \
  -d '{
    "schedule": "monthly",
    "full_dataset": true,
    "performance_threshold": 0.85
  }'
```

この手順により、既存モデルの性能を維持しながら効率的に新しいロゴを追加できます。

## クラスとデータセットの関係FAQ

### Q: データセットAでクラスXとYを学習後、データセットBでクラスXを再度学習する必要はあるの？

これは**学習方法と目的**によって異なります：

#### パターン1: 完全に新しいモデルを作成する場合

```bash
# 例：データセットBのみで新モデル作成
データセットA: [クラスX, クラスY, クラスZ] → モデルA
データセットB: [クラスX, クラスW, クラスV] → モデルB（新規作成）
```

**結果**: モデルBは**クラスX、W、Vのみ**検出可能
- クラスYとZは検出できない
- データセットAの学習内容は引き継がれない

#### パターン2: 転移学習で拡張する場合

```bash
# 例：モデルAをベースにデータセットBで転移学習
データセットA: [クラスX, クラスY, クラスZ] → モデルA
データセットB: [クラスW, クラスV] → モデルA' (転移学習)
```

**結果**: モデルA'は**全クラス（X、Y、Z、W、V）**を検出可能
- しかし**破滅的忘却**のリスクあり

#### パターン3: データセット統合で学習する場合（推奨）

```bash
# 例：データセットAとBを統合して学習
統合データセット: [クラスX（両方), クラスY, クラスZ, クラスW, クラスV] → モデルC
```

**結果**: 最も安定した性能を得られる

### 破滅的忘却（Catastrophic Forgetting）とは？

新しいクラスを学習する際に、**既存クラスの検出性能が劣化する現象**：

```bash
# 転移学習前後の性能比較例
クラスX (BANDAI):   95% → 88% (劣化)
クラスY (Nintendo): 92% → 85% (劣化)  
クラスZ (SONY):     90% → 82% (劣化)
クラスW (新企業A):   0% → 91% (新規)
クラスV (新企業B):   0% → 89% (新規)
```

### 最適な学習戦略

#### 戦略1: データセット統合（最も推奨）

```bash
# 1. 既存データセットに新クラスを追加
curl -X POST "http://localhost:8000/api/v1/training/datasets/trademark_logos_final/add-image" \
  -d '{
    "image_path": "data/logos/input/新企業A/logo1.jpg",
    "annotations": [{"class_name": "新企業A", "bbox": [100,50,300,150]}],
    "split": "train"
  }'

# 2. 統合データセットで学習
curl -X POST "http://localhost:8000/api/v1/training/start" \
  -d '{
    "model_name": "unified_model_v2",
    "dataset_name": "trademark_logos_final",
    "base_model": "models/trained/trademark_logos_final/weights/best.pt",
    "epochs": 50,
    "learning_rate": 0.0001
  }'
```

**メリット**:
- 全クラスの性能バランスが良い
- 破滅的忘却のリスクが最小
- データ量が多いほど精度向上

#### 戦略2: 段階的転移学習

```bash
# Phase 1: 特徴抽出層を凍結して新クラスのみ学習
curl -X POST "http://localhost:8000/api/v1/training/start" \
  -d '{
    "model_name": "incremental_phase1",
    "dataset_name": "new_company_logos",
    "base_model": "models/trained/trademark_logos_final/weights/best.pt",
    "epochs": 30,
    "learning_rate": 0.001,
    "freeze_layers": 15
  }'

# Phase 2: 全体をファインチューニング（統合データセット使用）
curl -X POST "http://localhost:8000/api/v1/training/start" \
  -d '{
    "model_name": "incremental_phase2", 
    "dataset_name": "unified_dataset",
    "base_model": "models/trained/incremental_phase1/weights/best.pt",
    "epochs": 20,
    "learning_rate": 0.0001,
    "freeze_layers": 0
  }'
```

#### 戦略3: 知識蒸留を使った学習

```bash
# 教師モデル（既存）と生徒モデル（新規）で知識を蒸留
curl -X POST "http://localhost:8000/api/v1/training/start-distillation" \
  -d '{
    "student_model_name": "distilled_model_v1",
    "teacher_model_path": "models/trained/trademark_logos_final/weights/best.pt",
    "new_dataset_name": "new_company_logos",
    "unified_dataset_name": "trademark_logos_final",
    "distillation_weight": 0.7,
    "classification_weight": 0.3
  }'
```

### 実践的な推奨事項

#### 新しいロゴを追加する場合の手順

1. **既存クラスのデータも含める**（推奨）
```bash
# 既存企業の追加画像も収集
data/logos/input/
├── BANDAI/              # 既存クラス
│   ├── existing1.jpg    # 既存画像
│   ├── new1.jpg         # 新規追加画像
│   └── new2.jpg
├── Nintendo/            # 既存クラス  
│   ├── existing1.jpg
│   └── new1.jpg
└── 新企業A/             # 新クラス
    ├── logo1.jpg
    └── logo2.jpg
```

2. **データバランスを確認**
```bash
curl "http://localhost:8000/api/v1/training/datasets/trademark_logos_final/balance-report"
```

3. **段階的な学習**
```bash
# まず少ないエポックでテスト
"epochs": 10

# 性能確認後に本格学習
"epochs": 50
```

#### クラスXを再学習すべきケース

✅ **再学習が必要な場合**:
- 既存クラスの検出精度を向上させたい
- 新しい背景・角度・サイズのデータを追加したい
- データのバランスを改善したい
- 破滅的忘却を防ぎたい

❌ **再学習が不要な場合**:
- 既存クラスの性能に満足している
- 計算リソースが限られている
- 新クラスとの関連性が低い

### 性能監視と調整

```bash
# 学習前後の性能比較
curl -X POST "http://localhost:8000/api/v1/training/compare-models" \
  -d '{
    "model_a": "models/old_model.pt",
    "model_b": "models/new_model.pt", 
    "test_dataset": "validation_set",
    "per_class_metrics": true
  }'

# クラス別性能確認
curl "http://localhost:8000/api/v1/training/class-performance-report?model=new_model"
```

**まとめ**: 新しいクラスを追加する際は、**既存クラスのデータも含めた統合学習**が最も安全で効果的です。

## ロゴ画像の選び方と種類

### Q: オフィシャルのきれいな画像だけでいいの？どんなパターンの画像が必要？

**答え**: **多様な状況の画像を混ぜることが重要**です。きれいな画像だけでは実用性が低くなります。

### 推奨する画像の種類とバランス

#### 1. ベース画像（20-30%）：高品質・クリア

```bash
# オフィシャル画像の例
data/logos/input/企業名/
├── official_logo_white_bg.png    # 白背景の公式ロゴ
├── official_logo_transparent.png # 透明背景の公式ロゴ
└── brand_guidelines.jpg          # ブランドガイドラインの画像
```

**特徴**:
- 背景がクリア（白・透明）
- ロゴが中央に配置
- 高解像度・高コントラスト
- 変形や歪みがない

**用途**: 
- 自動データセット生成の元画像
- 基準となる特徴量の学習

#### 2. 実世界画像（50-60%）：実用的な検出対象

```bash
# 実世界での使用例
data/logos/input/企業名/
├── product_package.jpg      # 商品パッケージ上のロゴ
├── storefront_sign.jpg      # 店舗看板
├── website_screenshot.jpg   # ウェブサイトのスクリーンショット
├── business_card.jpg        # 名刺
├── vehicle_branding.jpg     # 車両ラッピング
├── uniform_logo.jpg         # ユニフォーム
├── document_header.jpg      # 文書ヘッダー
└── advertisement_poster.jpg # 広告ポスター
```

**特徴**:
- 様々な背景（複雑・テクスチャあり）
- 異なるサイズ・角度・距離
- 照明条件の変化
- 部分的な隠れ・重なり

#### 3. 困難な条件の画像（20-30%）：ロバスト性向上

```bash
# チャレンジング画像の例
data/logos/input/企業名/
├── low_light_photo.jpg          # 暗い環境
├── blurry_motion.jpg            # ブレ・ボケ
├── partial_occlusion.jpg        # 部分的に隠れている
├── small_distant_logo.jpg       # 小さく遠くのロゴ
├── reflective_surface.jpg       # 反射面のロゴ
├── tilted_perspective.jpg       # 傾き・遠近法
├── multiple_logos_scene.jpg     # 複数ロゴが写っている
└── low_resolution.jpg           # 低解像度
```

### 具体的な収集方法

#### 方法1: 多様なソースから収集

```bash
# 推奨する収集先
1. 公式サイト・ブランドガイドライン  # 高品質ベース画像
2. Google画像検索                    # 実世界での使用例
3. 企業のSNS投稿                    # リアルな使用シーン
4. プレスリリース・ニュース記事      # メディア掲載例
5. 商品レビューサイト               # ユーザー投稿画像
6. ストリートビュー                 # 看板・店舗画像
```

#### 方法2: 撮影パターンの指針

```bash
# 撮影時のチェックリスト
✅ 背景の種類
  - 白・単色背景     (20%)
  - 複雑な背景       (40%) 
  - テクスチャ背景   (20%)
  - 屋外・自然背景   (20%)

✅ ロゴのサイズ
  - 大きい (画像の30%以上)     (30%)
  - 中程度 (画像の10-30%)      (40%)
  - 小さい (画像の10%未満)     (30%)

✅ 角度・向き
  - 正面              (50%)
  - 斜め・遠近法      (30%)
  - 回転・傾き        (20%)

✅ 照明条件
  - 明るい・均一      (40%)
  - 暗い・影あり      (30%)
  - 逆光・強い光      (30%)
```

### 自動データセット生成との組み合わせ

#### 元画像の質と生成結果の関係

```bash
# 高品質な元画像 → 多様な生成画像
元画像: official_logo_transparent.png (高品質)
↓ 自動生成
生成画像: 
├── logo_on_brick_wall.jpg      # 様々な背景に合成
├── logo_with_shadow.jpg        # 影・光効果追加
├── logo_small_corner.jpg       # サイズ・位置変更
├── logo_tilted_blur.jpg        # 変形・ボケ効果
└── logo_low_contrast.jpg       # コントラスト調整
```

**推奨バランス**:
```bash
元画像の種類:
├── 高品質公式画像    (3-5枚)   # 自動生成のベース用
├── 実世界使用例      (5-10枚)  # リアルな特徴学習用
└── 困難条件画像      (2-5枚)   # ロバスト性向上用

→ 自動生成で各元画像から20-30枚生成
→ 合計: 200-900枚の学習用画像
```

### 画像品質のガイドライン

#### ✅ 良い画像の例

```bash
# 高品質ベース画像
- 解像度: 512x512以上
- フォーマット: PNG（透明背景）、JPG（高品質）
- ロゴが明確に識別可能
- 文字が読める程度の解像度

# 実世界画像  
- 解像度: 300x300以上
- ロゴが画像の5%以上を占める
- 極端なボケ・ノイズがない
- 版権問題のない画像
```

#### ❌ 避けるべき画像

```bash
- 極端に小さいロゴ（画像の1%未満）
- 著しく低解像度（200x200未満）
- 著作権侵害の可能性がある画像
- ロゴが判別困難なほど変形
- 完全に隠れているロゴ
```

### 効率的な画像収集の実例

#### 企業ロゴの場合

```bash
# 例：スターバックスのロゴ収集
data/logos/input/STARBUCKS/
├── official_logo.png           # 公式サイトから
├── store_front.jpg            # Google Mapsから
├── coffee_cup.jpg             # 商品画像
├── business_document.pdf      # IR資料から
├── social_media_post.jpg      # 公式SNSから
├── news_article.jpg           # ニュース記事から
├── user_review_photo.jpg      # レビューサイトから
└── street_view.jpg            # ストリートビューから
```

#### 収集時の注意点

```bash
# 法的・倫理的考慮事項
1. 著作権・商標権の確認
2. 公正使用の範囲内での利用
3. 個人情報が写っていないか確認
4. 企業の公開画像の優先使用

# 技術的考慮事項  
1. メタデータの除去
2. 適切なファイル命名
3. 重複画像の除去
4. 品質チェック
```

### 最適な学習戦略

#### 段階的データ追加アプローチ

```bash
# Phase 1: 高品質画像でベースライン構築
curl -X POST "http://localhost:8000/api/v1/training/datasets/company_logos/generate-sample" \
  -d '{
    "classes": ["STARBUCKS"],
    "images_per_class": 20,
    "base_images_only": true,
    "quality_level": "high"
  }'

# Phase 2: 実世界画像を手動で追加（実際のAPI）
curl -X POST "http://localhost:8000/api/v1/training/datasets/company_logos/add-image" \
  -d '{
    "image_path": "data/logos/input/STARBUCKS/store_front.jpg",
    "annotations": [{"class_name": "STARBUCKS", "bbox": [100,50,300,150]}],
    "split": "train"
  }'

# Phase 3: 追加の自動生成データセット作成（実際のAPI）
curl -X POST "http://localhost:8000/api/v1/training/datasets/starbucks_additional/generate-sample" \
  -H "Content-Type: application/json" \
  -d '{
    "classes": ["STARBUCKS"],
    "images_per_class": 30
  }'
```

**結論**: **きれいな公式画像をベースに、実世界の多様な使用例を混ぜることが最も効果的**です。自動生成だけでは補えないリアルな特徴を学習できます。

## 実際のワークフロー推奨手順

### 💡 効率的な学習データ作成の流れ

現在のシステムでは**2つのアプローチ**が可能です。**組み合わせ使用**が最も効果的です：

#### アプローチ1: 自動生成メイン（推奨開始方法）

```bash
# Step 1: クリーンなロゴ画像を準備
data/logos/input/
├── 企業名A/
│   └── logo_clean.png        # 透明背景の高品質ロゴ
└── 企業名B/
    └── official_logo.png     # 公式ロゴ（背景なし）

# Step 2: 自動データセット生成（bbox設定不要）
curl -X POST "http://localhost:8000/api/v1/training/datasets/auto_logos/generate-sample" \
  -d '{
    "classes": ["企業名A", "企業名B"],
    "images_per_class": 30
  }'

# 結果: 60枚の学習用画像が自動生成（bbox自動設定済み）
# - 様々な背景に合成
# - ランダムサイズ・位置
# - 回転・変形・色調整
# - 自動アノテーション
```

#### アプローチ2: 実世界画像を手動追加（精度向上用）

```bash
# Step 3: 実世界画像を追加（bbox手動設定が必要）
data/logos/input/
├── 企業名A/
│   ├── store_photo.jpg       # 店舗看板の写真
│   ├── product_package.jpg   # 商品パッケージ
│   └── website_capture.jpg   # ウェブサイトキャプチャ

# Step 4: 手動でbbox設定して追加
curl -X POST "http://localhost:8000/api/v1/training/datasets/auto_logos/add-image" \
  -d '{
    "image_path": "data/logos/input/企業名A/store_photo.jpg",
    "annotations": [{"class_name": "企業名A", "bbox": [200,100,500,300]}],
    "split": "train"
  }'
```

#### 組み合わせ戦略の効果

```bash
# 最終的なデータセット構成例
データセット: auto_logos
├── 自動生成画像: 60枚      # ベースライン性能
│   ├── 企業名A: 30枚       # バリエーション豊富
│   └── 企業名B: 30枚       # bbox自動設定済み
└── 実世界画像: 10枚        # 実用性向上
    ├── 企業名A: 5枚        # 手動bbox設定
    └── 企業名B: 5枚        # リアルなシーン

→ 合計70枚で高精度モデル構築
```

### 🎯 どちらを選ぶべき？

#### 自動生成を選ぶ場合

✅ **メリット**:
- bbox設定が不要（効率的）
- 大量データを短時間で生成
- 一貫した品質
- 初心者でも簡単

✅ **適用場面**:
- 素早くプロトタイプを作りたい
- 大量のクラスを効率的に学習
- bbox設定の手間を省きたい
- 初期ベースライン構築

#### 手動アノテーションを選ぶ場合

✅ **メリット**:
- 実世界の正確なデータ
- より高い最終精度
- 特定シーンに特化可能
- 細かい制御が可能

✅ **適用場面**:
- 最高精度を目指したい
- 特定の使用シーンがある
- 実世界データが重要
- 時間をかけても良い

### 📋 実践的推奨手順

```bash
# Phase 1: 自動生成でベースライン構築（30分）
1. クリーンなロゴ画像を data/logos/input/ に配置
2. generate-sample API で自動データセット生成
3. 学習実行でベースライン性能確認

# Phase 2: 実世界画像で精度向上（必要に応じて）
1. 重要なシーンの実世界画像を収集
2. add-image API で手動bbox設定して追加
3. 再学習で性能向上を確認

# Phase 3: 継続改善
1. 誤検出が多いケースを分析
2. 該当シーンの画像を追加
3. 段階的に精度向上
```

### ⚡ クイックスタート例

```bash
# 1. ロゴ画像配置（1分）
mkdir -p data/logos/input/MyCompany
cp logo.png data/logos/input/MyCompany/

# 2. 自動データセット生成（5分）
curl -X POST "localhost:8000/api/v1/training/datasets/my_dataset/generate-sample?classes=MyCompany&images_per_class=25"

# 3. 学習開始（10-30分）
curl -X POST "localhost:8000/api/v1/training/start" \
  -d '{"model_name": "my_model", "dataset_name": "my_dataset", "epochs": 50}'

# 結果: 最短15分で動作するロゴ検出モデル完成
```

**推奨**: まず**自動生成**でベースラインを構築し、必要に応じて**実世界画像を手動追加**する段階的アプローチが最も効率的です。

## 実世界画像の処理方法

### Q: 実世界画像はロゴのところをトリミングするべき？それとも画像全体が必要？

**答え**: **画像全体を使用して、バウンディングボックスでロゴ位置を指定する**のが正解です。

### なぜ画像全体が必要なのか？

#### 物体検出（Object Detection）の仕組み

```bash
# 正しい方法：画像全体 + バウンディングボックス
画像全体 (1920x1080)
├── 背景情報: 店舗の外観、周辺環境
├── 他の要素: 看板の文字、建物、人など
└── ロゴ部分: [x:500, y:200, width:300, height:150] ← バウンディングボックス

# 間違った方法：ロゴ部分のみトリミング
トリミング画像 (300x150)
└── ロゴのみ ← 背景情報が失われる
```

#### YOLOv8が学習する内容

```bash
# 画像全体を使用した場合の学習内容
✅ ロゴの特徴（形状、色、文字）
✅ 背景との関係（コンテキスト）
✅ ロゴの位置予測能力
✅ 他の要素との区別能力
✅ スケール感（大きさの判断）

# トリミング画像の場合の学習内容  
❌ ロゴの特徴のみ
❌ 背景情報なし
❌ 位置予測能力なし
❌ 分類（Classification）になってしまう
```

### 実世界画像の正しい処理手順

#### 手順1: 画像全体を保持

```bash
# 元の画像（例：店舗の看板写真）
storefront_image.jpg (1920x1080)
├── 建物: 60% of image
├── 看板: 25% of image  
├── ロゴ: 8% of image    ← これが検出対象
└── その他: 7% of image
```

#### 手順2: ロゴの位置を特定

```bash
# バウンディングボックスの座標を取得
ロゴの位置:
- x_min: 500 (左上のx座標)
- y_min: 200 (左上のy座標)  
- x_max: 800 (右下のx座標)
- y_max: 350 (右下のy座標)
```

#### 手順3: アノテーション作成

```bash
# APIでの正しい画像追加方法
curl -X POST "http://localhost:8000/api/v1/training/datasets/company_logos/add-image" \
  -d '{
    "image_path": "data/logos/input/STARBUCKS/storefront_image.jpg",
    "annotations": [
      {
        "class_name": "STARBUCKS",
        "bbox": [500, 200, 800, 350],    # [x_min, y_min, x_max, y_max]
        "confidence": 1.0
      }
    ],
    "split": "train"
  }'
```

### 具体的な画像例の比較

#### ✅ 正しい例：画像全体を使用

```bash
# 店舗外観画像 (1200x800)
data/logos/input/STARBUCKS/store_photo.jpg
└── アノテーション: 
    - 画像サイズ: 1200x800
    - ロゴ位置: [300, 150, 600, 300]
    - 背景: 建物、道路、他の看板
    - 学習効果: ロゴ検出 + 位置予測 + 背景区別
```

#### ❌ 間違った例：ロゴのみトリミング

```bash
# ロゴ部分のみ (300x150)  
data/logos/input/STARBUCKS/logo_only.jpg
└── 問題:
    - 位置情報なし
    - 背景情報なし
    - スケール感なし
    - 分類問題になってしまう
```

### 様々なシーンでの適用例

#### 1. 商品パッケージの場合

```bash
# 商品全体の写真
product_package.jpg (800x600)
├── 商品の形状・色
├── パッケージデザイン
├── ロゴ位置: [200, 100, 400, 180]
└── その他のテキスト・要素

→ 学習効果: 商品上でのロゴ検出能力
```

#### 2. ウェブサイトスクリーンショットの場合

```bash
# ページ全体のスクリーンショット
website_screenshot.jpg (1920x1080)
├── ヘッダー部分
├── ナビゲーション
├── ロゴ位置: [50, 20, 200, 80]
└── その他のコンテンツ

→ 学習効果: デジタル環境でのロゴ検出能力
```

#### 3. 文書・名刺の場合

```bash
# 文書全体
business_card.jpg (600x400)
├── テキスト情報
├── レイアウト
├── ロゴ位置: [450, 50, 580, 120]
└── 連絡先情報

→ 学習効果: 文書内でのロゴ検出能力
```

### バウンディングボックスの作成方法

#### 方法1: 手動で座標を測定

```bash
# 画像編集ソフトを使用
1. 画像をPhotoshop/GIMP等で開く
2. ロゴ部分を矩形選択
3. 座標をメモ（x, y, width, height）
4. [x_min, y_min, x_max, y_max]に変換
```

#### 方法2: アノテーションツール使用

```bash
# LabelImg等のツール使用
1. LabelImgでプロジェクト作成
2. 画像を読み込み
3. 矩形でロゴを囲む
4. クラス名を指定
5. YOLO形式で出力
```

#### 方法3: プログラムで自動検出

```python
# OpenCVを使った半自動アノテーション
import cv2
import numpy as np

def find_logo_bbox(image_path, template_path):
    """テンプレートマッチングでロゴ位置を検出"""
    img = cv2.imread(image_path)
    template = cv2.imread(template_path)
    
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if max_val > 0.8:  # 信頼度閾値
        h, w = template.shape[:2]
        x_min, y_min = max_loc
        x_max, y_max = x_min + w, y_min + h
        return [x_min, y_min, x_max, y_max]
    
    return None
```

### サイズ・品質の考慮事項

#### 画像サイズのガイドライン

```bash
# 推奨画像サイズ
✅ 最小: 640x640    # YOLOv8の標準入力サイズ
✅ 推奨: 1280x1280  # 高精度検出用
✅ 最大: 1920x1920  # 計算コスト許容範囲

# ロゴサイズの目安
✅ 最小: 32x32ピクセル以上
✅ 推奨: 画像の5-30%を占める
✅ 最大: 画像の80%まで
```

#### 品質要件

```bash
# 画像品質
✅ フォーカス: ロゴ部分がピンボケしていない
✅ 解像度: ロゴの文字が識別可能
✅ コントラスト: 背景とロゴが区別できる
✅ 圧縮: 過度なJPEG圧縮なし

# バウンディングボックス精度
✅ 精密さ: ロゴ境界に正確にフィット
✅ 余白: 最小限の余白（2-5ピクセル）
✅ 欠損: ロゴの一部が枠外に出ない
```

### 効率的なワークフロー

#### 大量画像処理の場合

```bash
# 1. 画像の一括前処理
find data/logos/input/ -name "*.jpg" -exec python resize_images.py {} \;

# 2. 自動アノテーション（可能な場合）
python auto_annotate.py --input_dir data/logos/input/STARBUCKS/ --template official_logo.png

# 3. 手動確認・修正
python review_annotations.py --dataset_dir datasets/company_logos/

# 4. データセットに一括追加
python batch_add_images.py --dataset_name company_logos --input_dir annotated_images/
```

### 自動データセット生成との使い分け

```bash
# 元画像の役割分担

1. 高品質公式画像（トリミング済み）
   └── 自動データセット生成の元画像として使用
   └── 様々な背景に合成される

2. 実世界画像（全体画像 + アノテーション）
   └── リアルな検出学習データとして使用
   └── 実際の使用シーンでの特徴学習

→ 両方を組み合わせることで最適な学習効果
```

**まとめ**: **実世界画像は画像全体を保持し、バウンディングボックスでロゴ位置を指定**することで、位置予測と背景区別能力を含む完全な物体検出モデルが構築できます。

## 開発ガイド

### プロジェクト構造

```
logo-detection-api/
├── src/
│   ├── api/          # FastAPI エンドポイント
│   ├── core/         # 検出エンジン・バッチ処理
│   ├── models/       # Pydantic スキーマ
│   └── utils/        # ユーティリティ
├── tests/            # テストコード
├── docker/           # Docker 設定
├── models/           # 学習済みモデル
├── data/             # データセット・ロゴ画像
│   ├── logos/
│   │   ├── input/    # 元画像配置場所
│   │   └── samples/  # サンプル画像
│   └── datasets/     # 生成されたデータセット
└── logs/             # ログファイル
```

### 拡張方法

#### カスタムモデルの追加

```python
# src/core/detection_engine.py の修正例
def load_custom_model(self, model_path: str):
    """カスタムモデルの読み込み"""
    self.model = YOLO(model_path)
    # ロゴ固有の後処理を追加
```

#### 新しいエンドポイントの追加

```python
# src/api/endpoints/new_endpoint.py
from fastapi import APIRouter

router = APIRouter()

@router.post("/new-feature")
async def new_feature():
    # 新機能の実装
    pass
```

## ライセンス

[ライセンス情報をここに記載]

## 貢献

[コントリビューションガイドラインをここに記載]

## サポート

問題や質問がある場合は、以下の方法でお問い合わせください：

- **Issues**: GitHub Issues
- **Email**: [サポートメールアドレス]
- **Documentation**: [ドキュメントURL]

---

**注意**: このシステムは数百万枚の画像処理を想定していますが、実際の運用時は段階的にスケールアップし、システムリソースを適切に監視することを推奨します。