-- inspection_batchesテーブルの作成
-- 既存のテーブルがある場合はスキップ

CREATE TABLE IF NOT EXISTS public.inspection_batches (
    batch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    mode VARCHAR(50) NOT NULL CHECK (mode IN ('seller', 'path')),
    model_name VARCHAR(255) NOT NULL,
    confidence_threshold DECIMAL(3,2) DEFAULT 0.5,
    max_detections INTEGER DEFAULT 10,
    total_items INTEGER DEFAULT 0,
    processed_items INTEGER DEFAULT 0,
    detected_items INTEGER DEFAULT 0,
    failed_items INTEGER DEFAULT 0,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    processing_time_seconds INTEGER GENERATED ALWAYS AS (
        CASE 
            WHEN end_time IS NOT NULL THEN 
                EXTRACT(EPOCH FROM (end_time - start_time))::INTEGER
            ELSE NULL
        END
    ) STORED,
    seller_id VARCHAR(255),
    base_path TEXT,
    include_subdirs BOOLEAN DEFAULT true,
    error_message TEXT,
    average_confidence DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- インデックスの作成
CREATE INDEX IF NOT EXISTS idx_inspection_batches_status ON public.inspection_batches(status);
CREATE INDEX IF NOT EXISTS idx_inspection_batches_seller_id ON public.inspection_batches(seller_id);
CREATE INDEX IF NOT EXISTS idx_inspection_batches_start_time ON public.inspection_batches(start_time DESC);
CREATE INDEX IF NOT EXISTS idx_inspection_batches_created_at ON public.inspection_batches(created_at DESC);

-- コメントの追加
COMMENT ON TABLE public.inspection_batches IS '検査バッチ管理テーブル';
COMMENT ON COLUMN public.inspection_batches.batch_id IS 'バッチID（UUID）';
COMMENT ON COLUMN public.inspection_batches.status IS 'バッチ状態（pending/running/completed/failed/cancelled）';
COMMENT ON COLUMN public.inspection_batches.mode IS '検査モード（seller/path）';
COMMENT ON COLUMN public.inspection_batches.processing_time_seconds IS '処理時間（秒）自動計算';

-- inspection_statisticsテーブルの作成（統計情報用）
CREATE TABLE IF NOT EXISTS public.inspection_statistics (
    date DATE PRIMARY KEY,
    total_inspections INTEGER DEFAULT 0,
    total_detections INTEGER DEFAULT 0,
    unique_sellers INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- inventoryテーブルに検査結果カラムを追加（存在しない場合のみ）
DO $$
BEGIN
    -- model_usedカラム
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'inventory' AND column_name = 'model_used') THEN
        ALTER TABLE public.inventory ADD COLUMN model_used TEXT;
    END IF;
    
    -- last_processed_atカラム
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'inventory' AND column_name = 'last_processed_at') THEN
        ALTER TABLE public.inventory ADD COLUMN last_processed_at TIMESTAMP WITH TIME ZONE;
    END IF;
    
    -- detectionsカラム
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'inventory' AND column_name = 'detections') THEN
        ALTER TABLE public.inventory ADD COLUMN detections JSONB DEFAULT '[]';
    END IF;
    
    -- confidence_scoresカラム
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'inventory' AND column_name = 'confidence_scores') THEN
        ALTER TABLE public.inventory ADD COLUMN confidence_scores DECIMAL(3,2)[] DEFAULT '{}';
    END IF;
    
    -- labelsカラム
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'inventory' AND column_name = 'labels') THEN
        ALTER TABLE public.inventory ADD COLUMN labels TEXT[] DEFAULT '{}';
    END IF;
    
    -- detectedカラム
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'inventory' AND column_name = 'detected') THEN
        ALTER TABLE public.inventory ADD COLUMN detected BOOLEAN DEFAULT false;
    END IF;
    
    -- batch_idカラム
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'inventory' AND column_name = 'batch_id') THEN
        ALTER TABLE public.inventory ADD COLUMN batch_id UUID;
    END IF;
END $$;

-- インデックスの追加（inventoryテーブル）
CREATE INDEX IF NOT EXISTS idx_inventory_model_used ON public.inventory(model_used);
CREATE INDEX IF NOT EXISTS idx_inventory_last_processed ON public.inventory(last_processed_at);
CREATE INDEX IF NOT EXISTS idx_inventory_detected ON public.inventory(detected);
CREATE INDEX IF NOT EXISTS idx_inventory_batch_id ON public.inventory(batch_id);

-- 検査結果を高速に検索するための複合インデックス
CREATE INDEX IF NOT EXISTS idx_inventory_seller_model ON public.inventory(seller_id, model_used);
CREATE INDEX IF NOT EXISTS idx_inventory_detection_search ON public.inventory(seller_id, detected, last_processed_at DESC);

GRANT ALL PRIVILEGES ON TABLE public.inspection_batches TO admin;
GRANT ALL PRIVILEGES ON TABLE public.inspection_statistics TO admin;