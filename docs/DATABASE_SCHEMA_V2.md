# Logo Detection System - データベース構造 V2

## 概要
Logo Detection System V2では、検査結果を直接`inventory`テーブルに保存する設計に変更しました。これにより、商品情報と検査結果が一元管理され、データの整合性が向上します。

## データベース接続情報
- **ホスト**: localhost
- **ポート**: 5432
- **データベース名**: logo_detection
- **ユーザー**: admin
- **パスワード**: Inagaki1

## テーブル構造

### 1. inventory（メインテーブル）
商品在庫情報と検査結果を管理する中心的なテーブルです。

| カラム名 | 型 | 説明 | デフォルト値 |
|---------|-----|------|------------|
| id | serial | 主キー | AUTO INCREMENT |
| asin | text | Amazon標準識別番号 | NOT NULL |
| seller_id | text | セラーID | NOT NULL |
| image_url | text | 画像URL | NULL |
| model_used | text | 使用した検出モデル名 | NULL |
| last_processed_at | timestamp with time zone | 最終処理日時 | '2000-01-01 09:00:00+09' |
| sku | text | 在庫管理単位 | NOT NULL |
| has_image_url | boolean | 画像URL有無 | (image_url IS NOT NULL) |
| detections | jsonb | 検出結果詳細 | '[]' |
| confidence_scores | numeric(3,2)[] | 信頼度スコア配列 | '{}' |
| labels | text[] | 検出ラベル配列 | '{}' |
| detected | boolean | 検出有無 | false |
| batch_id | uuid | 検査バッチID | NULL |

**インデックス**:
- `inventory_pkey`: id (PRIMARY KEY)
- `idx_inventory_seller_id`: seller_id
- `idx_inventory_asin`: asin
- `idx_inventory_seller_asin`: seller_id, asin
- `idx_inventory_batch_id`: batch_id
- `idx_inventory_detected`: detected
- `idx_inventory_last_processed`: last_processed_at
- `idx_inventory_has_image_url`: has_image_url
- `idx_inventory_detections`: detections (GIN index)

### 2. inspection_batches（検査バッチ管理）
検査の実行単位を管理するテーブルです。

| カラム名 | 型 | 説明 | 制約 |
|---------|-----|------|------|
| batch_id | UUID | バッチID（主キー） | PRIMARY KEY, DEFAULT gen_random_uuid() |
| status | VARCHAR(50) | 実行状態 | NOT NULL, DEFAULT 'pending' |
| mode | VARCHAR(50) | 検査モード | NOT NULL |
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
| base_path | TEXT | 基準パス（path mode時） | - |
| include_subdirs | BOOLEAN | サブディレクトリ含む | DEFAULT true |
| error_message | TEXT | エラーメッセージ | - |
| processing_time_seconds | DECIMAL(10,2) | 処理時間（秒） | - |
| average_confidence | DECIMAL(3,2) | 平均信頼度 | - |
| created_at | TIMESTAMP | 作成日時 | DEFAULT CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | 更新日時 | DEFAULT CURRENT_TIMESTAMP |

**検査モード**:
- `seller`: セラーID指定で検査（DBから取得）
- `path`: 絶対パス指定で検査（ローカルファイル）
- `user`: ユーザーID指定（未実装）
- `all`: 全件検査（未実装）

## 処理フロー

### 1. セラーID指定での検査
```sql
-- 1. 検査対象を取得
SELECT id, seller_id, asin, sku, image_url
FROM inventory
WHERE seller_id = 'A3OBH97MEO1982'
AND has_image_url = true
ORDER BY last_processed_at ASC
LIMIT 1000;

-- 2. 検査結果を更新
UPDATE inventory 
SET 
    model_used = 'yolov8n',
    last_processed_at = CURRENT_TIMESTAMP,
    detections = '[{"logo_name": "BANDAI", "confidence": 0.95}]'::jsonb,
    confidence_scores = ARRAY[0.95],
    labels = ARRAY['BANDAI'],
    detected = true,
    batch_id = 'batch-uuid-here'
WHERE seller_id = 'A3OBH97MEO1982' AND asin = 'B001234567';
```

### 2. 絶対パス指定での検査
パス指定の場合、inventoryテーブルにレコードが存在しない可能性があるため、検査結果は別途管理が必要です。

## ビュー

### active_inspection_batches
現在実行中または待機中の検査バッチを表示します。

### inspection_results_summary
バッチごとの検査結果サマリーを表示します。

### brand_detection_stats
ブランド別の検出統計を表示します。

## 便利な関数

### get_unprocessed_count(seller_id)
特定セラーの未処理画像数を取得します。
```sql
SELECT get_unprocessed_count('A3OBH97MEO1982');
```

### get_batch_progress(batch_id)
バッチの進捗情報を取得します。
```sql
SELECT * FROM get_batch_progress('batch-uuid-here');
```

## パフォーマンス最適化

### 大量データ処理時の考慮事項
1. **バッチ更新**: `executemany`を使用して複数レコードを一括更新
2. **インデックス**: 検索・更新頻度の高いカラムにインデックスを設定済み
3. **処理優先順位**: `last_processed_at`の古い順に処理
4. **並列処理**: バッチサイズとワーカー数を調整可能

### 推奨設定
- **CPU環境**: バッチサイズ 8-16、並列数 8-12
- **GPU環境**: バッチサイズ 32-64、並列数 2-4

## 移行手順

既存のシステムから移行する場合：

1. inventoryテーブルに必要なカラムが存在することを確認
2. inspection_batchesテーブルを作成
3. インデックスを作成
4. ビューと関数を作成

```bash
psql -h localhost -U admin -d logo_detection -f postgresql-setup/02_update_inventory_table.sql
```

## 注意事項

- `inventory`テーブルの既存データは保持されます
- 検査結果は`inventory`テーブルを直接更新するため、トランザクション管理が重要
- パス指定モードでは、inventoryに存在しない画像も処理可能ですが、結果の保存先は要検討