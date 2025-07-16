# 画像検査機能

既存のロゴ検出APIに追加された画像検査機能の説明書です。

## 概要

商品画像に対してロゴ検出を実行し、結果をDBに保存する機能です。大量の画像を効率的に処理するため、バッチ処理とGPU負荷管理機能を搭載しています。

## 機能

### 1. 画像検査API

- **検査対象**: DBから取得した商品情報（現在はダミーデータ）
- **画像パス形式**: `basePath/{sellerId}/{asinの末尾1文字}/{ASIN}.png`
- **GPU負荷管理**: 負荷率を0.1〜1.0で調整可能
- **モデル選択**: 実行時に任意のモデルを選択
- **結果保存**: 検査結果をDBに保存（現在はダミー実装）

### 2. 実行モード

#### 個別指定モード
- セラーIDとユーザーID（オプション）を指定
- 特定の出品者やユーザーの商品のみを検査

#### 全件実行モード
- 全レコードを対象に検査
- 処理上限数の設定が可能
- デフォルトで全件処理（チェックボックスで制御）

### 3. 管理UI

Webベースの管理画面で以下の操作が可能：
- 検査の開始（モード選択、パラメータ設定）
- 実行状況のモニタリング
- 統計情報の確認
- バッチのキャンセル

## 使用方法

### APIエンドポイント

#### 検査開始
```
POST /api/v1/inspection/start
```

リクエストボディ例（個別指定モード）：
```json
{
  "mode": "individual",
  "model_name": "logo_detector_v32_fixed",
  "gpu_load_rate": 0.8,
  "seller_id": "SELLER_001",
  "user_id": "USER_0001",
  "max_items": 100,
  "confidence_threshold": 0.5,
  "max_detections": 10
}
```

リクエストボディ例（全件実行モード）：
```json
{
  "mode": "bulk",
  "model_name": "logo_detector_v32_fixed",
  "gpu_load_rate": 0.8,
  "process_all": true,
  "confidence_threshold": 0.5,
  "max_detections": 10
}
```

#### ステータス確認
```
GET /api/v1/inspection/status/{batch_id}
GET /api/v1/inspection/status  # 全バッチのステータス
```

#### バッチキャンセル
```
POST /api/v1/inspection/cancel/{batch_id}
```

#### ダッシュボード情報
```
GET /api/v1/inspection/dashboard
```

### 管理UI

ブラウザで以下にアクセス：
```
http://localhost:8000/ui/inspection
```

## 実装詳細

### ファイル構成

```
src/
├── models/
│   └── inspection_schemas.py    # データモデル定義
├── core/
│   └── inspection_engine.py     # 検査処理エンジン
├── api/endpoints/
│   └── inspection.py           # APIエンドポイント
├── db/
│   └── dummy_db.py            # ダミーDB実装
└── templates/
    └── inspection_ui.html      # 管理UI
```

### GPU負荷管理

`GPULoadManager`クラスが負荷率に応じて処理間隔を調整：
- 負荷率1.0: 最大速度で処理
- 負荷率0.5: 処理間に適度な待機時間
- 負荷率0.1: 最小負荷で処理

### 非同期処理

検査は非同期で実行され、即座にバッチIDが返されます。進捗はステータスAPIで確認できます。

## 今後の実装予定

1. **実際のDB接続**
   - PostgreSQL/MySQLへの接続
   - 商品情報の取得
   - 検査結果の永続化

2. **画像ストレージ連携**
   - S3/GCSからの画像取得
   - キャッシュ機能

3. **詳細な統計機能**
   - 時系列グラフ
   - ブランド別検出率
   - セラー別分析

4. **通知機能**
   - 検査完了通知
   - エラー通知
   - 定期レポート

## トラブルシューティング

### GPU関連のエラー
- CUDAが利用できない場合は自動的にCPUモードに切り替わります
- メモリ不足の場合はバッチサイズを調整してください

### 画像が見つからない
- 現在はダミー実装のため、実際の画像ファイルは不要です
- 本番環境では画像パスの設定を確認してください

### 処理が遅い
- GPU負荷率を上げる
- より高性能なモデル（yolov8n → yolov8s）を使用
- バッチサイズを調整

## サンプルコード

### Python
```python
import requests

# 検査開始
response = requests.post(
    "http://localhost:8000/api/v1/inspection/start",
    json={
        "mode": "individual",
        "model_name": "logo_detector_v32_fixed",
        "gpu_load_rate": 0.8,
        "seller_id": "SELLER_001",
        "max_items": 100
    }
)
batch_id = response.json()["batch_id"]

# ステータス確認
status = requests.get(
    f"http://localhost:8000/api/v1/inspection/status/{batch_id}"
).json()
print(f"Progress: {status['progress']}%")
```

### JavaScript
```javascript
// 検査開始
const response = await fetch('/api/v1/inspection/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        mode: 'bulk',
        model_name: 'logo_detector_v32_fixed',
        gpu_load_rate: 0.8,
        process_all: true
    })
});
const { batch_id } = await response.json();

// 進捗モニタリング
const checkProgress = async () => {
    const status = await fetch(`/api/v1/inspection/status/${batch_id}`).json();
    console.log(`Progress: ${status.progress}%`);
    
    if (status.status !== 'completed') {
        setTimeout(checkProgress, 2000);
    }
};
checkProgress();
```