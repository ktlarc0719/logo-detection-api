#!/usr/bin/env python3
"""
データベーステーブルの存在確認スクリプト
"""

import asyncio
import asyncpg
from src.core.config import get_settings

async def check_tables():
    """テーブルの存在を確認"""
    settings = get_settings()
    
    # データベース接続
    conn = await asyncpg.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        database=settings.db_name
    )
    
    try:
        # inspection_batchesテーブルの確認
        print("=== Checking inspection_batches table ===")
        query = """
        SELECT EXISTS (
            SELECT 1 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'inspection_batches'
        );
        """
        exists = await conn.fetchval(query)
        print(f"inspection_batches table exists: {exists}")
        
        if exists:
            # カラム情報を取得
            columns_query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public' 
            AND table_name = 'inspection_batches'
            ORDER BY ordinal_position;
            """
            columns = await conn.fetch(columns_query)
            print("\nColumns in inspection_batches:")
            for col in columns:
                print(f"  - {col['column_name']} ({col['data_type']}) {'' if col['is_nullable'] == 'YES' else 'NOT NULL'}")
        
        # inventoryテーブルの検査関連カラムの確認
        print("\n=== Checking inventory table columns ===")
        inv_columns_query = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public' 
        AND table_name = 'inventory'
        AND column_name IN ('model_used', 'last_processed_at', 'detections', 
                           'confidence_scores', 'labels', 'detected', 'batch_id')
        ORDER BY column_name;
        """
        inv_columns = await conn.fetch(inv_columns_query)
        print("Inspection-related columns in inventory:")
        for col in inv_columns:
            print(f"  - {col['column_name']} ({col['data_type']})")
        
        # レコード数の確認
        if exists:
            count_query = "SELECT COUNT(*) FROM inspection_batches;"
            count = await conn.fetchval(count_query)
            print(f"\nNumber of records in inspection_batches: {count}")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(check_tables())