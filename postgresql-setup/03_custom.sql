DROP TABLE IF EXISTS public.inventory CASCADE;

-- inventoryテーブル作成
CREATE TABLE public.inventory (
    id SERIAL PRIMARY KEY,
    asin TEXT NOT NULL,
  	sku TEXT NOT NULL,
  	title TEXT NOT NULL,
    seller_id TEXT NOT NULL,
    image_url TEXT NULL,
  	has_image_url BOOLEAN NOT NULL GENERATED ALWAYS AS (image_url IS NOT NULL) STORED,
  
  	-- 制約
  	CONSTRAINT inventory_seller_id_asin_unique 
      	UNIQUE (seller_id, asin)
);

-- コメント追加
COMMENT ON TABLE public.inventory IS 'Product inventory with image information';
COMMENT ON COLUMN public.inventory.has_image_url IS 'Auto-generated: true if image_url is not null, false otherwise';

DROP TABLE IF EXISTS public.inspection_results CASCADE;
CREATE TABLE public.inspection_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID NOT NULL,
    inventory_id INTEGER NOT NULL,
    detected BOOLEAN NOT NULL DEFAULT false,
    model_used TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 検出結果のJSON
    detections JSONB DEFAULT '[]',
    confidence_scores DECIMAL(3,2)[] DEFAULT '{}',
    labels TEXT[] DEFAULT '{}',
    
    -- エラー情報
    error_message TEXT,
    
    -- 制約
    CONSTRAINT inspection_results_inventory_fk 
        FOREIGN KEY (inventory_id) REFERENCES public.inventory(id) ON DELETE CASCADE,
    CONSTRAINT inspection_results_inventory_model_unique 
        UNIQUE (inventory_id, model_used)
);

-- コメント追加
COMMENT ON TABLE public.inspection_results IS 'Image inspection results for each model';
COMMENT ON COLUMN public.inspection_results.batch_id IS 'Batch identifier for grouping related inspections. If same model is used, new batch_id is overwritten.';
COMMENT ON COLUMN public.inspection_results.detected IS 'Whether any objects were detected';
COMMENT ON COLUMN public.inspection_results.detections IS 'Detailed detection results in JSON format';

-- INDEX

-- seller_id検索用（最大10万件の一括取得）
CREATE INDEX idx_inventory_seller_id ON public.inventory (seller_id);

-- seller_id + has_image_url での検索用
CREATE INDEX idx_inventory_seller_image ON public.inventory (seller_id, has_image_url);

-- seller_id + id での範囲検索用（ページネーション対応）
CREATE INDEX idx_inventory_seller_id_pagination ON public.inventory (seller_id, id);

-- SKU検索用
CREATE INDEX idx_inventory_sku ON public.inventory (sku);

--inspection_resultsテーブル
-- 外部キー用インデックス（JOINに必須）
CREATE INDEX idx_inspection_inventory_id ON public.inspection_results (inventory_id);

-- inventory_id + model_used 検索用（メインの検索パターン）
CREATE INDEX idx_inspection_inventory_model ON public.inspection_results (inventory_id, model_used);

-- model_used検索用
CREATE INDEX idx_inspection_model_used ON public.inspection_results (model_used);

-- seller_id + model_used での検索用（inventoryとのJOIN前提）
CREATE INDEX idx_inspection_seller_model ON public.inspection_results (model_used, created_at DESC);

-- batch_id検索用
CREATE INDEX idx_inspection_batch_id ON public.inspection_results (batch_id);

-- 検出成功レコード検索用
CREATE INDEX idx_inspection_detected ON public.inspection_results (detected, model_used) 
WHERE detected = true;

-- エラー追跡用
CREATE INDEX idx_inspection_errors ON public.inspection_results (model_used, created_at DESC) 
WHERE error_message IS NOT NULL;

-- カバリングインデックス（SELECT性能向上）
CREATE INDEX idx_inspection_covering ON public.inspection_results (inventory_id, model_used) 
INCLUDE (detected, detections, confidence_scores, labels, created_at);


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