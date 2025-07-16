# 新規検査機能の仕様書

## 1. 基本方針
- 既存のベース機能は担保しつつ、必要に応じて大幅変更を許容
- DBは作成済み（postgresql-setup/03_custom.sqlに実行済みSQL保存）
- 記載のsqlについては、参考程度のため、必要に応じて変更すること

## 2. UI仕様変更

### 2.1 検査モード
- **削除対象**: 「ユーザーID指定検査」「全数検査」
- **残すモード**: 「セラーID指定検査」→ モード名を `seller_id` に変更
- **モード選択UI**: 1つしか残らないため選択機能は削除

### 2.2 検査対象選択（ラジオボタン）
- 🔘 **未検査のみ**（デフォルト）
- ⚪ **全検査**

### 2.3 入力フォーム
- セラーID：テキスト入力フィールド
- 検査開始ボタン

## 3. 処理フロー

### 3.1 データ取得処理
```sql
-- 未検査のみの場合
SELECT i.*
FROM inventory i
WHERE i.seller_id = :seller_id
  AND i.has_image_url = true
  AND NOT EXISTS (
    SELECT 1 FROM inspection_results ir 
    WHERE ir.inventory_id = i.id 
      AND ir.model_used = :model_used
  )

-- 全検査の場合
SELECT i.*
FROM inventory i
WHERE i.seller_id = :seller_id
  AND i.has_image_url = true
```

### 3.2 画像パス特定ロジック
```
画像保存ルール:
/mnt/c/03_amazon_images/{セラーID}/{ASINの末尾1文字}/{ASIN}.png

例: seller_id:A3OBH97MEO1982 ASIN:B0FSWEGOE2 → /mnt/c/03_amazon_images/A3OBH97MEO1982/2/B0FSWEGOE2.png
```

### 3.3 効率化処理
1. `/mnt/c/03_amazon_images/{セラーID}/` 配下の全画像ファイルを事前取得
2. サブディレクトリ（0-9, A-Z）を含む全ファイルリストを作成
3. 取得したレコードのASINと照合して存在する画像のみを検査対象とする

### 3.4 検査実行
1. `inspection_batches`テーブルに新しいバッチを作成
2. 対象画像に対して検査を実行
3. 進捗状況を`inspection_batches`に更新

### 3.5 結果保存
```sql
-- inspection_resultsへの保存（UPSERT）
INSERT INTO inspection_results (
    batch_id, inventory_id, detected, model_used, detections, 
    confidence_scores, labels, error_message
) VALUES (...)
ON CONFLICT (inventory_id, model_used) 
DO UPDATE SET
    batch_id = EXCLUDED.batch_id,
    detected = EXCLUDED.detected,
    detections = EXCLUDED.detections,
    confidence_scores = EXCLUDED.confidence_scores,
    labels = EXCLUDED.labels,
    error_message = EXCLUDED.error_message,
    created_at = NOW();
```

## 4. 必要なテーブル構造確認

### 4.1 inspection_batches
```sql
-- 既存テーブルがあるかチェックし、なければ作成
CREATE TABLE IF NOT EXISTS inspection_batches (
    batch_id UUID PRIMARY KEY,
    description TEXT,
    seller_id TEXT,
    model_used TEXT,
    total_items INTEGER DEFAULT 0,
    processed_items INTEGER DEFAULT 0,
    detected_items INTEGER DEFAULT 0,
    error_items INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);
```

## 5. 確認が必要な点

### 5.1 UI/UX確認事項
1. **検査モード名**: 残す1つのモードの適切な名前は？
2. **進捗表示**: 検査中の進捗をリアルタイム表示する？
3. **結果表示**: 検査完了後の結果表示方法は？
4. **エラーハンドリング**: ファイルが見つからない場合の処理は？

### 5.2 ビジネスロジック確認事項
1. **重複処理**: 同じバッチで同じ画像を複数回処理する可能性は？
2. **検査失敗時**: 一部の画像で検査が失敗した場合の全体処理継続は？
3. **バッチサイズ**: 大量データの場合、バッチを分割する必要は？

## 5. キューイング機能

### 5.1 キュー処理フロー
1. **検査開始ボタン押下** → キューに追加
2. **キューワーカー** → 順次処理（1つずつ）
3. **処理中は他のリクエストを待機状態に**

## 6. ファイルキャッシュ機能

### 6.1 簡単なキャッシュ実装
```python
# メモリキャッシュまたは一時ファイルでの実装
cache = {}  # {seller_id: {file_list, timestamp}}

def get_image_files(seller_id):
    cache_key = seller_id
    cache_ttl = 300  # 5分間キャッシュ
    
    if cache_key in cache:
        cached_data = cache[cache_key]
        if time.time() - cached_data['timestamp'] < cache_ttl:
            return cached_data['files']
    
    # キャッシュなし/期限切れの場合はファイルスキャン
    files = scan_image_directory(seller_id)
    cache[cache_key] = {
        'files': files,
        'timestamp': time.time()
    }
    return files
```

## 7. 実装推奨事項

### 7.1 API設計
```
- 既存のAPI設計を踏襲し、必要なものを作ってください。
```

## 8. Claude Codeへの最終指示

上記の仕様で以下を実装してください：

### 8.1 削除・変更対象
- 「ユーザーID指定検査」「全数検査」モードの完全削除
- 「セラーID指定検査」を`seller_id`モードに名前変更。これは将来的にモードが増える可能性があるため、inspection_batchesに残すだけでいい。
- モード選択UIの削除（1つしか残らないため）

### 8.2 新規実装
1. **キューイング機能**: 
   - 検査リクエストをキューに蓄積
   - 順次処理（同時実行なし）
   - キュー状況の表示

2. **ファイルキャッシュ機能**:
   - セラーIDごとの画像ファイルリスト
   - 5分間程度の簡単なメモリキャッシュ

3. **効率的な画像ファイル処理**:
   - 事前にディレクトリ全体をスキャン
   - ASINとの照合で存在確認

### 8.3 保持・改良対象
- 既存の非同期処理機能
- 検査対象選択（未検査のみ/全検査）
- 進捗表示機能
- 結果保存のUPSERT処理

### 8.4 技術要件
- 最大10万件のレコード処理に対応
- `/mnt/c/03_amazon_images/{セラーID}/{ASINの末尾1文字}/{ASIN}.png` のパス構造
- PostgreSQLの既存スキーマとの整合性維持

既存コードベースとの整合性を保ちつつ、上記仕様を満たす実装をお願いします。