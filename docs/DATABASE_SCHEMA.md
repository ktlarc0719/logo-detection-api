# Logo Detection System - データベース構造

## 概要
Logo Detection Systemは、PostgreSQLデータベース（`logo_detection`）を使用して、画像検査の実行管理と結果の保存を行います。本ドキュメントでは、システムで使用される主要なテーブルとその構造について説明します。

## データベース接続情報
- **ホスト**: localhost
- **ポート**: 5432
- **データベース名**: logo_detection
- **ユーザー**: admin
- **パスワード**: Inagaki1

## テーブル構造

### 1. inspection_batches（検査バッチ管理）
検査の実行単位を管理するテーブルです。

| カラム名 | 型 | 説明 | 制約 |
|---------|-----|------|------|
| batch_id | UUID | バッチID（主キー） | PRIMARY KEY, DEFAULT gen_random_uuid() |
| status | VARCHAR(50) | 実行状態 | NOT NULL, DEFAULT 'pending' |
| mode | VARCHAR(50) | 検査モード（seller/path/user/all） | NOT NULL |
| model_name | VARCHAR(255) | 使用モデル名 | NOT NULL |
| confidence_threshold | DECIMAL(3,2) | 信頼度閾値 | DEFAULT 0.5 |
| max_detections | INTEGER | 最大検出数 | DEFAULT 10 |
| total_items | INTEGER | 総アイテム数 | DEFAULT 0 |
| processed_items | INTEGER | 処理済みアイテム数 | DEFAULT 0 |
| detected_items | INTEGER | 検出ありアイテム数 | DEFAULT 0 |
| failed_items | INTEGER | 失敗アイテム数 | DEFAULT 0 |
| start_time | TIMESTAMP | 開始時刻 | DEFAULT CURRENT_TIMESTAMP |
| end_time | TIMESTAMP | 終了時刻 | - |
| seller_id | VARCHAR(255) | セラーID（seller mode時） | - |
| user_id | VARCHAR(255) | ユーザーID（user mode時） | - |
| base_path | TEXT | 基準パス（path mode時） | - |
| include_subdirs | BOOLEAN | サブディレクトリ含む | DEFAULT true |
| error_message | TEXT | エラーメッセージ | - |
| processing_time_seconds | DECIMAL(10,2) | 処理時間（秒） | - |
| average_confidence | DECIMAL(3,2) | 平均信頼度 | - |
| created_at | TIMESTAMP | 作成日時 | DEFAULT CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | 更新日時 | DEFAULT CURRENT_TIMESTAMP |

**ステータス値**:
- `pending`: 待機中
- `running`: 実行中
- `completed`: 完了
- `failed`: 失敗
- `cancelled`: キャンセル

**インデックス**:
- `idx_inspection_batches_status`: status
- `idx_inspection_batches_seller_id`: seller_id
- `idx_inspection_batches_created_at`: created_at DESC

### 2. inspection_results（検査結果）
各画像の検査結果を保存するテーブルです。

| カラム名 | 型 | 説明 | 制約 |
|---------|-----|------|------|
| result_id | UUID | 結果ID（主キー） | PRIMARY KEY, DEFAULT gen_random_uuid() |
| batch_id | UUID | バッチID | NOT NULL, FOREIGN KEY |
| item_id | VARCHAR(255) | アイテムID | - |
| seller_id | VARCHAR(255) | セラーID | - |
| asin | VARCHAR(255) | ASIN | - |
| user_id | VARCHAR(255) | ユーザーID | - |
| image_path | TEXT | 画像パス | NOT NULL |
| detected | BOOLEAN | 検出有無 | NOT NULL, DEFAULT false |
| detection_count | INTEGER | 検出数 | DEFAULT 0 |
| model_used | VARCHAR(255) | 使用モデル | - |
| processing_time | DECIMAL(10,3) | 処理時間（秒） | - |
| detections | JSONB | 検出詳細（JSON） | DEFAULT '[]' |
| confidence_scores | DECIMAL(3,2)[] | 信頼度スコア配列 | DEFAULT '{}' |
| labels | TEXT[] | ラベル配列 | DEFAULT '{}' |
| error_message | TEXT | エラーメッセージ | - |
| created_at | TIMESTAMP | 作成日時 | DEFAULT CURRENT_TIMESTAMP |

**detections JSONBフォーマット**:
```json
[
  {
    "logo_name": "BANDAI",
    "confidence": 0.95,
    "bbox": [100, 200, 300, 400],
    "class": "toy"
  }
]
```

**インデックス**:
- `idx_inspection_results_batch_id`: batch_id
- `idx_inspection_results_seller_id`: seller_id
- `idx_inspection_results_detected`: detected
- `idx_inspection_results_created_at`: created_at DESC
- `idx_inspection_results_detections`: detections (GIN index)

### 3. inspection_statistics（検査統計）
日次の検査統計を保存するテーブルです。

| カラム名 | 型 | 説明 | 制約 |
|---------|-----|------|------|
| stat_id | UUID | 統計ID（主キー） | PRIMARY KEY, DEFAULT gen_random_uuid() |
| date | DATE | 日付 | NOT NULL, DEFAULT CURRENT_DATE, UNIQUE |
| total_inspections | INTEGER | 総検査数 | DEFAULT 0 |
| total_detections | INTEGER | 総検出数 | DEFAULT 0 |
| unique_sellers | INTEGER | ユニークセラー数 | DEFAULT 0 |
| unique_brands | INTEGER | ユニークブランド数 | DEFAULT 0 |
| average_processing_time | DECIMAL(10,3) | 平均処理時間 | - |
| brand_counts | JSONB | ブランド別カウント | DEFAULT '{}' |
| seller_stats | JSONB | セラー別統計 | DEFAULT '{}' |
| hourly_stats | JSONB | 時間別統計 | DEFAULT '{}' |
| created_at | TIMESTAMP | 作成日時 | DEFAULT CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | 更新日時 | DEFAULT CURRENT_TIMESTAMP |

### 4. inventory（既存テーブル）
商品在庫情報を管理する既存のテーブルです。検査対象の画像情報はこのテーブルから取得されます。

## ビュー

### active_inspection_batches
現在実行中または待機中の検査バッチを表示するビューです。

```sql
SELECT 
    batch_id,
    status,
    mode,
    model_name,
    total_items,
    processed_items,
    progress_percentage,
    start_time,
    elapsed_seconds
FROM active_inspection_batches;
```

### inspection_summary
検査バッチのサマリー情報を表示するビューです。

```sql
SELECT 
    batch_id,
    status,
    mode,
    seller_id,
    base_path,
    total_results,
    detected_count,
    avg_processing_time,
    created_at
FROM inspection_summary;
```

## 使用例

### 1. 新しい検査バッチの作成
```sql
INSERT INTO inspection_batches (mode, model_name, seller_id, confidence_threshold)
VALUES ('seller', 'yolov8n', 'A3OBH97MEO1982', 0.7)
RETURNING batch_id;
```

### 2. 検査結果の保存
```sql
INSERT INTO inspection_results (
    batch_id, seller_id, asin, image_path, 
    detected, detection_count, detections, 
    confidence_scores, labels, processing_time
)
VALUES (
    'batch-uuid-here',
    'A3OBH97MEO1982',
    'B001234567',
    '/mnt/c/03_amazon_images/A3OBH97MEO1982/image001.jpg',
    true,
    2,
    '[{"logo_name": "BANDAI", "confidence": 0.95, "bbox": [100, 200, 300, 400]}]'::jsonb,
    ARRAY[0.95, 0.87],
    ARRAY['BANDAI', 'TOMICA'],
    0.234
);
```

### 3. 実行中のバッチ状態を更新
```sql
UPDATE inspection_batches 
SET 
    status = 'running',
    processed_items = processed_items + 1,
    detected_items = detected_items + 1
WHERE batch_id = 'batch-uuid-here';
```

### 4. セラーIDに基づく画像パスの取得
```sql
-- inventoryテーブルから特定セラーの画像パスを取得
SELECT 
    seller_id,
    asin,
    image_path
FROM inventory
WHERE seller_id = 'A3OBH97MEO1982'
LIMIT 1000;
```

### 5. 統計情報の更新
```sql
-- 本日の統計を更新
INSERT INTO inspection_statistics (date, total_inspections, total_detections)
VALUES (CURRENT_DATE, 100, 45)
ON CONFLICT (date) 
DO UPDATE SET 
    total_inspections = inspection_statistics.total_inspections + EXCLUDED.total_inspections,
    total_detections = inspection_statistics.total_detections + EXCLUDED.total_detections;
```

## パフォーマンス考慮事項

### 大量データ処理時の推奨事項
1. **バッチ処理**: 一度に大量のレコードを挿入する場合は、トランザクション内でバッチ処理を行う
2. **インデックス**: 検索頻度の高いカラムにはインデックスが設定済み
3. **JSONB**: 検出結果はJSONB型で保存され、GINインデックスにより高速検索が可能
4. **パーティション**: 将来的にデータ量が増大した場合は、日付によるパーティション分割を検討

### 接続プール設定
- 最小接続数: 5
- 最大接続数: 20
- タイムアウト: 30秒

## メンテナンス

### 定期的なVACUUM
```sql
VACUUM ANALYZE inspection_results;
VACUUM ANALYZE inspection_batches;
```

### 古いデータのアーカイブ
```sql
-- 90日以上前の結果をアーカイブ
CREATE TABLE inspection_results_archive AS 
SELECT * FROM inspection_results 
WHERE created_at < CURRENT_DATE - INTERVAL '90 days';

DELETE FROM inspection_results 
WHERE created_at < CURRENT_DATE - INTERVAL '90 days';
```