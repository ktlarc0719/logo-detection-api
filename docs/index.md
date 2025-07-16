# Logo Detection API Documentation

## 概要

Logo Detection APIは、高性能なロゴ検出システムです。YOLOv8をベースに、Webインターフェース、バッチ処理、機械学習パイプラインを統合しています。

## ドキュメント一覧

### システムガイド

- **[UI Navigation Guide](UI_NAVIGATION_GUIDE.md)** - 統合ダッシュボードとナビゲーションの使い方
- **[ML System Guide](ML_SYSTEM_GUIDE.md)** - 機械学習システムの完全ガイド
- **[Inspection Feature](INSPECTION_FEATURE.md)** - 画像検査機能の詳細

### デプロイメント

- **[Deployment](DEPLOYMENT.md)** - 基本的なデプロイメント手順
- **[Docker Hub Deployment](DOCKER_HUB_DEPLOYMENT.md)** - Docker Hubを使用したデプロイ
- **[VPS Deployment](VPS_DEPLOYMENT.md)** - VPSへのデプロイメント
- **[Deployment Strategies](DEPLOYMENT_STRATEGIES.md)** - デプロイメント戦略

### 開発者向け

- **[Local Test Setup](LOCAL_TEST_SETUP.md)** - ローカル開発環境のセットアップ
- **[Build Performance](BUILD_PERFORMANCE.md)** - ビルドパフォーマンスの最適化
- **[Training](training.md)** - モデルトレーニングガイド
- **[Google Colab](GOOGLE_COLAB.md)** - Google Colabでの利用方法

### API仕様

- **[URL Batch API](URL_BATCH_API.md)** - URL一括処理APIの仕様

### その他

- **[Manual Git Push](MANUAL_GIT_PUSH.md)** - Git手動プッシュ手順
- **[VPS Docker Exec](VPS_DOCKER_EXEC.md)** - VPS上でのDocker実行

## クイックリンク

- **メインREADME**: [/README.md](../README.md)
- **APIドキュメント**: http://localhost:8000/docs
- **ダッシュボード**: http://localhost:8000/

## システム構成

```
logo-detection-api/
├── src/                    # ソースコード
│   ├── api/               # API実装
│   ├── core/              # コアロジック
│   ├── models/            # データモデル
│   └── utils/             # ユーティリティ
├── templates/             # HTMLテンプレート
├── static/                # 静的ファイル
├── models/                # 学習済みモデル
├── datasets/              # データセット
├── docs/                  # ドキュメント
└── tests/                 # テストコード
```

## 主要機能

1. **ロゴ検出**
   - 単一画像検出
   - バッチ処理
   - URLからの検出

2. **機械学習**
   - データセット検証
   - モデルトレーニング
   - 性能評価
   - 可視化

3. **画像検査**
   - 商品画像の自動検査
   - GPU負荷管理
   - 統計分析

4. **Web UI**
   - 統合ダッシュボード
   - 各機能への簡単アクセス
   - リアルタイム監視

## サポート

問題が発生した場合は、以下を確認してください：

1. [トラブルシューティング](UI_NAVIGATION_GUIDE.md#トラブルシューティング)
2. [ローカルテストセットアップ](LOCAL_TEST_SETUP.md)
3. APIドキュメント（/docs）