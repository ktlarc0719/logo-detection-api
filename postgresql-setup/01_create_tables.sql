-- Logo Detection System Database Schema
-- Database: logo_detection

-- ====================================
-- 1. インベントリテーブル（既存）
-- ====================================
-- inventory テーブルは既に存在すると仮定

-- ====================================
-- 2. 検査バッチテーブル
-- ====================================
CREATE TABLE IF NOT EXISTS inspection_batches (
    batch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status TEXT NOT NULL DEFAULT 'pending',
    mode TEXT NOT NULL, -- 'seller', 'path', 'user', 'all'
    model_name TEXT NOT NULL,
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
    seller_id TEXT,
    user_id TEXT,
    base_path TEXT,
    include_subdirs BOOLEAN DEFAULT true,
    
    -- メタデータ
    error_message TEXT,
    processing_time_seconds DECIMAL(10,2),
    average_confidence DECIMAL(3,2),
    
    CONSTRAINT check_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'))
);

-- インデックス
CREATE INDEX idx_inspection_batches_status ON inspection_batches(status);
CREATE INDEX idx_inspection_batches_seller_id ON inspection_batches(seller_id);
CREATE INDEX idx_inspection_batches_created_at ON inspection_batches(created_at DESC);

-- ====================================
-- 3. 検査結果テーブル
-- ====================================
CREATE TABLE IF NOT EXISTS inspection_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID NOT NULL REFERENCES inspection_batches(batch_id) ON DELETE CASCADE,
    item_id VARCHAR(255),
    seller_id VARCHAR(255),
    asin VARCHAR(255),
    user_id VARCHAR(255),
    image_path TEXT NOT NULL,
    detected BOOLEAN NOT NULL DEFAULT false,
    detection_count INTEGER DEFAULT 0,
    model_used VARCHAR(255),
    processing_time DECIMAL(10,3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 検出結果のJSON
    detections JSONB DEFAULT '[]',
    confidence_scores DECIMAL(3,2)[] DEFAULT '{}',
    labels TEXT[] DEFAULT '{}',
    
    -- エラー情報
    error_message TEXT,
    
    CONSTRAINT fk_batch_id FOREIGN KEY (batch_id) REFERENCES inspection_batches(batch_id)
);

-- インデックス
CREATE INDEX idx_inspection_results_batch_id ON inspection_results(batch_id);
CREATE INDEX idx_inspection_results_seller_id ON inspection_results(seller_id);
CREATE INDEX idx_inspection_results_detected ON inspection_results(detected);
CREATE INDEX idx_inspection_results_created_at ON inspection_results(created_at DESC);
CREATE INDEX idx_inspection_results_detections ON inspection_results USING GIN (detections);

-- ====================================
-- 4. 検査統計テーブル（集計用）
-- ====================================
CREATE TABLE IF NOT EXISTS inspection_statistics (
    stat_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date DATE NOT NULL DEFAULT CURRENT_DATE,
    total_inspections INTEGER DEFAULT 0,
    total_detections INTEGER DEFAULT 0,
    unique_sellers INTEGER DEFAULT 0,
    unique_brands INTEGER DEFAULT 0,
    average_processing_time DECIMAL(10,3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 詳細統計（JSON）
    brand_counts JSONB DEFAULT '{}',
    seller_stats JSONB DEFAULT '{}',
    hourly_stats JSONB DEFAULT '{}'
);

-- ユニーク制約（1日1レコード）
CREATE UNIQUE INDEX idx_inspection_statistics_date ON inspection_statistics(date);

-- ====================================
-- 5. 更新日時の自動更新トリガー
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

CREATE TRIGGER update_inspection_statistics_updated_at BEFORE UPDATE
    ON inspection_statistics FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ====================================
-- 6. ビュー定義
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
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - start_time)) as elapsed_seconds
FROM inspection_batches
WHERE status IN ('pending', 'running')
ORDER BY created_at DESC;

-- 検査サマリービュー
CREATE OR REPLACE VIEW inspection_summary AS
SELECT 
    ib.batch_id,
    ib.status,
    ib.mode,
    ib.seller_id,
    ib.base_path,
    COUNT(ir.result_id) as total_results,
    SUM(CASE WHEN ir.detected THEN 1 ELSE 0 END) as detected_count,
    AVG(ir.processing_time) as avg_processing_time,
    ib.created_at
FROM inspection_batches ib
LEFT JOIN inspection_results ir ON ib.batch_id = ir.batch_id
GROUP BY ib.batch_id, ib.status, ib.mode, ib.seller_id, ib.base_path, ib.created_at
ORDER BY ib.created_at DESC;

-- ====================================
-- 7. 初期データ
-- ====================================
-- 本日の統計レコードを作成
INSERT INTO inspection_statistics (date) 
VALUES (CURRENT_DATE) 
ON CONFLICT (date) DO NOTHING;