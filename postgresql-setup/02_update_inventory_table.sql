-- Logo Detection System - Inventory Table Updates
-- inventoryテーブルは既に存在すると仮定

-- ====================================
-- 1. 検査バッチテーブル（簡略版）
-- ====================================
CREATE TABLE IF NOT EXISTS inspection_batches (
    batch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    mode VARCHAR(50) NOT NULL, -- 'seller', 'path', 'user', 'all'
    model_name VARCHAR(255) NOT NULL,
    confidence_threshold DECIMAL(3,2) DEFAULT 0.5,
    max_detections INTEGER DEFAULT 10,
    total_items INTEGER DEFAULT 0,
    processed_items INTEGER DEFAULT 0,
    detected_items INTEGER DEFAULT 0,
    failed_items INTEGER DEFAULT 0,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 検査パラメータ
    seller_id VARCHAR(255),
    user_id VARCHAR(255),
    base_path TEXT,
    include_subdirs BOOLEAN DEFAULT true,
    
    -- メタデータ
    error_message TEXT,
    processing_time_seconds DECIMAL(10,2),
    average_confidence DECIMAL(3,2),
    
    CONSTRAINT check_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'))
);

-- インデックス
CREATE INDEX IF NOT EXISTS idx_inspection_batches_status ON inspection_batches(status);
CREATE INDEX IF NOT EXISTS idx_inspection_batches_seller_id ON inspection_batches(seller_id);
CREATE INDEX IF NOT EXISTS idx_inspection_batches_created_at ON inspection_batches(created_at DESC);

-- ====================================
-- 2. inventoryテーブルのインデックス追加
-- ====================================
-- 検査処理のパフォーマンス向上のため
CREATE INDEX IF NOT EXISTS idx_inventory_seller_id ON inventory(seller_id);
CREATE INDEX IF NOT EXISTS idx_inventory_asin ON inventory(asin);
CREATE INDEX IF NOT EXISTS idx_inventory_seller_asin ON inventory(seller_id, asin);
CREATE INDEX IF NOT EXISTS idx_inventory_batch_id ON inventory(batch_id);
CREATE INDEX IF NOT EXISTS idx_inventory_detected ON inventory(detected);
CREATE INDEX IF NOT EXISTS idx_inventory_last_processed ON inventory(last_processed_at);
CREATE INDEX IF NOT EXISTS idx_inventory_has_image_url ON inventory(has_image_url);

-- GINインデックス for JSONB
CREATE INDEX IF NOT EXISTS idx_inventory_detections ON inventory USING GIN (detections);

-- ====================================
-- 3. 更新日時の自動更新トリガー
-- ====================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_inspection_batches_updated_at BEFORE UPDATE
    ON inspection_batches FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ====================================
-- 4. ビュー定義
-- ====================================

-- アクティブな検査バッチビュー
CREATE OR REPLACE VIEW active_inspection_batches AS
SELECT 
    batch_id,
    status,
    mode,
    model_name,
    total_items,
    processed_items,
    CASE 
        WHEN total_items > 0 THEN ROUND((processed_items::DECIMAL / total_items) * 100, 2)
        ELSE 0
    END as progress_percentage,
    start_time,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - start_time)) as elapsed_seconds,
    seller_id,
    user_id,
    base_path
FROM inspection_batches
WHERE status IN ('pending', 'running')
ORDER BY created_at DESC;

-- 検査結果サマリービュー
CREATE OR REPLACE VIEW inspection_results_summary AS
SELECT 
    i.batch_id,
    COUNT(*) as total_items,
    COUNT(CASE WHEN i.detected THEN 1 END) as detected_items,
    COUNT(DISTINCT i.seller_id) as unique_sellers,
    ARRAY_AGG(DISTINCT unnest(i.labels)) as unique_labels,
    MAX(i.last_processed_at) as last_update
FROM inventory i
WHERE i.batch_id IS NOT NULL
GROUP BY i.batch_id;

-- 検出ブランド統計ビュー
CREATE OR REPLACE VIEW brand_detection_stats AS
SELECT 
    label,
    COUNT(*) as detection_count,
    COUNT(DISTINCT seller_id) as seller_count,
    COUNT(DISTINCT batch_id) as batch_count
FROM inventory
CROSS JOIN LATERAL unnest(labels) as label
WHERE detected = true
GROUP BY label
ORDER BY detection_count DESC;

-- ====================================
-- 5. 便利な関数
-- ====================================

-- 特定セラーの未処理画像数を取得
CREATE OR REPLACE FUNCTION get_unprocessed_count(p_seller_id TEXT)
RETURNS INTEGER AS $$
BEGIN
    RETURN (
        SELECT COUNT(*)
        FROM inventory
        WHERE seller_id = p_seller_id
        AND has_image_url = true
        AND last_processed_at < '2001-01-01'::timestamp
    );
END;
$$ LANGUAGE plpgsql;

-- バッチの進捗情報を取得
CREATE OR REPLACE FUNCTION get_batch_progress(p_batch_id UUID)
RETURNS TABLE(
    total_items INTEGER,
    processed_items INTEGER,
    detected_items INTEGER,
    progress_percentage DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        b.total_items,
        b.processed_items,
        b.detected_items,
        CASE 
            WHEN b.total_items > 0 THEN ROUND((b.processed_items::DECIMAL / b.total_items) * 100, 2)
            ELSE 0
        END as progress_percentage
    FROM inspection_batches b
    WHERE b.batch_id = p_batch_id;
END;
$$ LANGUAGE plpgsql;