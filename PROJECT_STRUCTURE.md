# Project Structure

## ディレクトリ構成

```
logo-detection-api/
├── src/                        # アプリケーションソースコード
│   ├── api/                   # FastAPI関連
│   │   ├── endpoints/         # APIエンドポイント
│   │   │   ├── batch_detection.py
│   │   │   ├── single_detection.py
│   │   │   ├── health.py
│   │   │   ├── model_management.py
│   │   │   ├── training.py
│   │   │   ├── logo_management.py
│   │   │   ├── download_images.py
│   │   │   ├── system_management.py
│   │   │   ├── url_batch_detection.py
│   │   │   ├── inspection.py
│   │   │   ├── ml_system.py
│   │   │   └── dataset_splitter.py
│   │   └── main.py           # FastAPIアプリケーション
│   ├── core/                  # コアビジネスロジック
│   │   ├── config.py         # 設定管理
│   │   ├── detection_engine.py
│   │   ├── training_engine.py
│   │   ├── inspection_engine.py
│   │   ├── ml_dataset_validator.py
│   │   ├── ml_training_engine.py
│   │   ├── ml_model_visualizer.py
│   │   └── ml_model_validator.py
│   ├── models/                # Pydanticモデル
│   │   ├── schemas.py
│   │   ├── training_schemas.py
│   │   ├── inspection_schemas.py
│   │   └── ml_system_schemas.py
│   ├── utils/                 # ユーティリティ
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   └── file_utils.py
│   └── db/                    # データベース関連
│       └── dummy_db.py
│
├── templates/                  # HTMLテンプレート
│   ├── dashboard.html         # メインダッシュボード
│   ├── inspection_ui.html     # 検査UI
│   └── ml_system_ui.html      # MLシステムUI
│
├── static/                     # 静的ファイル
│   ├── index.html            # 単一画像検出UI
│   └── batch.html            # バッチ処理UI
│
├── models/                     # 学習済みモデル
│   ├── general_model_v3.pt
│   ├── trademark_model_v2.2.pt
│   └── yolov8n.pt
│
├── datasets/                   # データセット
│   └── (各種データセット)
│
├── docs/                       # ドキュメント
│   ├── index.md              # ドキュメントインデックス
│   ├── UI_NAVIGATION_GUIDE.md
│   ├── ML_SYSTEM_GUIDE.md
│   ├── INSPECTION_FEATURE.md
│   ├── DEPLOYMENT.md
│   ├── DOCKER_HUB_DEPLOYMENT.md
│   ├── VPS_DEPLOYMENT.md
│   ├── DEPLOYMENT_STRATEGIES.md
│   ├── LOCAL_TEST_SETUP.md
│   ├── BUILD_PERFORMANCE.md
│   ├── training.md
│   ├── GOOGLE_COLAB.md
│   ├── URL_BATCH_API.md
│   ├── MANUAL_GIT_PUSH.md
│   └── VPS_DOCKER_EXEC.md
│
├── scripts/                    # 各種スクリプト
│   └── (デプロイメント、ユーティリティスクリプト)
│
├── tests/                      # テストコード
│   └── (ユニットテスト、統合テスト)
│
├── docker/                     # Docker関連ファイル
│   └── (Dockerfileなど)
│
├── nginx/                      # Nginx設定
│   └── nginx.conf
│
├── logs/                       # ログファイル（gitignore）
├── runs/                       # YOLO実行結果（gitignore）
├── data/                       # データファイル（gitignore）
│
├── .env.example               # 環境変数サンプル
├── .gitignore                 # Git除外設定
├── Dockerfile                 # Dockerイメージ定義
├── docker-compose.yml         # Docker Compose設定
├── docker-compose.production.yml
├── docker-compose.vps.yml
├── requirements.txt           # Python依存関係
├── Makefile                   # ビルド・デプロイコマンド
├── setup.sh                   # セットアップスクリプト
├── logging.conf               # ロギング設定
├── README.md                  # プロジェクトREADME
└── PROJECT_STRUCTURE.md       # このファイル
```

## 主要コンポーネント

### 1. API層 (`src/api/`)
- FastAPIを使用したRESTful API
- 各機能ごとにエンドポイントを分離
- OpenAPI/Swagger自動生成

### 2. ビジネスロジック層 (`src/core/`)
- 検出エンジン
- トレーニングエンジン
- 検査エンジン
- MLシステムコンポーネント

### 3. データモデル層 (`src/models/`)
- Pydanticスキーマ定義
- リクエスト/レスポンスモデル
- 内部データ構造

### 4. UI層 (`templates/`, `static/`)
- HTMLテンプレート（サーバーサイドレンダリング）
- 静的HTML（クライアントサイド）
- Bootstrap/JavaScriptベースのUI

### 5. 設定・ユーティリティ
- 環境変数ベースの設定
- ロギング、メトリクス
- ファイル操作ユーティリティ

## 開発ガイドライン

1. **新機能追加時**
   - `src/api/endpoints/`に新しいエンドポイントを作成
   - `src/core/`にビジネスロジックを実装
   - `src/models/`に必要なスキーマを定義

2. **UI追加時**
   - 統合UIの場合は`templates/`に追加
   - スタンドアロンUIの場合は`static/`に追加
   - ダッシュボードにナビゲーションを追加

3. **ドキュメント**
   - 新機能は`docs/`にドキュメントを追加
   - README.mdの更新も忘れずに

4. **テスト**
   - `tests/`にユニットテストを追加
   - 統合テストも考慮

## クリーンアーキテクチャ

このプロジェクトは以下の原則に従っています：

- **関心の分離**: 各層は独立して変更可能
- **依存性の逆転**: ビジネスロジックはフレームワークに依存しない
- **テスタビリティ**: 各コンポーネントは独立してテスト可能
- **拡張性**: 新機能の追加が容易