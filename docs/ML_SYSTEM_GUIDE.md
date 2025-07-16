# 機械学習システム

ロゴ検出APIに統合された機械学習システムの説明書です。

## 概要

Google Colabで成功した機械学習ワークフローをローカル環境に移行し、統合されたWebベースのUIを提供します。データセット検証、モデルトレーニング、可視化、検証のすべての機能を一つのシステムで管理できます。

## 機能

### 1. データセット検証
- YOLOフォーマットのデータセットを検証
- 画像とラベルの整合性チェック
- クラス分布の分析
- データセットの問題点を自動検出

### 2. モデルトレーニング
- **トレーニングモード**
  - 新規学習（ゼロから）
  - 転移学習（既存モデルをベースに）
  - 継続学習（中断したトレーニングを再開）
- **モデルアーキテクチャ**
  - YOLOv8n（最速）
  - YOLOv8s（バランス）
  - YOLOv8m（高精度）
  - YOLOv8l（より高精度）
  - YOLOv8x（最高精度）
- **リアルタイム進捗モニタリング**
- **GPU負荷管理**

### 3. モデル可視化
- トレーニング履歴のグラフ表示
- 混同行列
- PR曲線、F1曲線
- クラス別パフォーマンス分析

### 4. モデル検証
- テストデータセットでの性能評価
- 詳細なメトリクス計算（Accuracy、Precision、Recall、F1）
- エラー分析と可視化
- 予測結果の保存

## 使用方法

### Web UI
ブラウザで以下にアクセス：
```
http://localhost:8000/ui/ml
```

### APIエンドポイント

#### システムステータス
```
GET /api/v1/ml/status
```
システム全体のステータス（GPU、CPU、メモリ使用率など）を取得

#### データセット検証
```
POST /api/v1/ml/dataset/validate
```
リクエストボディ例：
```json
{
  "dataset_path": "/path/to/dataset",
  "check_images": true,
  "check_labels": true,
  "validate_format": true
}
```

#### トレーニング開始
```
POST /api/v1/ml/training/start
```
リクエストボディ例（新規学習）：
```json
{
  "mode": "full",
  "dataset_path": "/path/to/dataset",
  "model_architecture": "yolov8s",
  "epochs": 100,
  "batch_size": 16,
  "learning_rate": 0.01
}
```

リクエストボディ例（転移学習）：
```json
{
  "mode": "transfer",
  "dataset_path": "/path/to/dataset",
  "model_architecture": "yolov8s",
  "base_model_path": "/path/to/base/model.pt",
  "epochs": 50,
  "batch_size": 16,
  "learning_rate": 0.001
}
```

#### トレーニングステータス確認
```
GET /api/v1/ml/training/status/{training_id}
GET /api/v1/ml/training/status  # 全トレーニング
```

#### モデル可視化
```
POST /api/v1/ml/model/visualize
```
リクエストボディ例：
```json
{
  "model_path": "/path/to/model.pt",
  "include_confusion_matrix": true,
  "include_pr_curve": true,
  "include_f1_curve": true,
  "include_training_history": true,
  "include_class_metrics": true
}
```

#### モデル検証
```
POST /api/v1/ml/model/validate
```
リクエストボディ例：
```json
{
  "model_path": "/path/to/model.pt",
  "test_dataset_path": "/path/to/test/dataset",
  "confidence_threshold": 0.25,
  "iou_threshold": 0.45,
  "save_predictions": true,
  "analyze_errors": true
}
```

## ディレクトリ構成

```
ml_outputs/
├── trainings/           # トレーニング出力
│   └── {project_name}/
│       └── {experiment_name}/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           ├── results.csv
│           ├── confusion_matrix.png
│           └── training_config.yaml
├── validations/         # 検証結果
│   └── {model_name}_{timestamp}/
│       ├── confusion_matrix.png
│       └── error_samples/
└── visualizations/      # 可視化結果
    └── {model_name}_{timestamp}/
        ├── training_curves.png
        └── class_metrics.png
```

## 推奨ワークフロー

### 1. 新しいモデルのトレーニング
1. データセットを準備（YOLOフォーマット）
2. Web UIでデータセット検証を実行
3. 問題がなければトレーニングを開始
4. リアルタイムで進捗を監視
5. トレーニング完了後、モデルを検証

### 2. 既存モデルの改善
1. 現在のモデルでテストデータセットを検証
2. エラー分析結果を確認
3. 問題のあるクラスのデータを追加
4. 転移学習でモデルを再トレーニング
5. 改善を確認

### 3. 継続的な改善サイクル
1. 本番環境でのエラーパターンを収集
2. エラー画像をデータセットに追加
3. データセットを再検証
4. 転移学習で更新
5. A/Bテストで性能比較

## トラブルシューティング

### GPUが認識されない
- CUDAが正しくインストールされているか確認
- PyTorchのGPU版がインストールされているか確認
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### メモリ不足エラー
- バッチサイズを小さくする
- より小さいモデルアーキテクチャを使用
- 画像サイズ（imgsz）を小さくする

### トレーニングが遅い
- GPUを使用しているか確認
- データセットのキャッシュを有効化
- バッチサイズを大きくする（メモリが許す範囲で）

## パフォーマンス最適化のヒント

### データセット準備
- バランスの取れたクラス分布を維持
- 高品質な画像を使用
- 適切なデータ拡張を適用

### トレーニング設定
- 学習率のスケジューリングを活用
- 早期停止（patience）を適切に設定
- 定期的にチェックポイントを保存

### 推論最適化
- 適切な信頼度しきい値を設定
- バッチ推論を活用
- モデルの量子化を検討

## サンプルコード

### Python
```python
import requests

# データセット検証
response = requests.post(
    "http://localhost:8000/api/v1/ml/dataset/validate",
    json={
        "dataset_path": "dataset",
        "check_images": True,
        "check_labels": True,
        "validate_format": True
    }
)
print(response.json())

# トレーニング開始
response = requests.post(
    "http://localhost:8000/api/v1/ml/training/start",
    json={
        "mode": "full",
        "dataset_path": "dataset",
        "model_architecture": "yolov8s",
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 0.01
    }
)
training_id = response.json()["training_id"]

# ステータス確認
import time
while True:
    response = requests.get(
        f"http://localhost:8000/api/v1/ml/training/status/{training_id}"
    )
    status = response.json()
    print(f"Progress: {status['progress']}%")
    
    if status['status'] in ['completed', 'failed']:
        break
    
    time.sleep(5)
```

## 今後の拡張予定

1. **AutoML機能**
   - ハイパーパラメータの自動調整
   - アーキテクチャの自動選択

2. **分散トレーニング**
   - 複数GPUサポート
   - クラスタ環境での実行

3. **モデル管理**
   - バージョン管理
   - A/Bテスト機能
   - モデルレジストリ

4. **高度な分析**
   - アクティブラーニング
   - データドリフト検出
   - モデル説明可能性

5. **統合機能**
   - CI/CDパイプライン連携
   - クラウドストレージ連携
   - MLflowとの統合