"""
データベース操作ユーティリティ
5000万件の大量データ処理を考慮した実装
"""

from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncpg
from datetime import datetime

from src.db.connection import db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseUtils:
    """データベース操作ユーティリティクラス"""
    
    @staticmethod
    async def fetch_paginated(
        query: str,
        page_size: int = 1000,
        params: Optional[list] = None,
        order_by: Optional[str] = None
    ) -> AsyncGenerator[List[Dict], None]:
        """
        ページネーションを使用してデータを取得
        大量データの処理に適している
        """
        offset = 0
        
        # ORDER BY句を追加（ページネーションには必須）
        if order_by:
            query += f" ORDER BY {order_by}"
        
        while True:
            paginated_query = f"{query} LIMIT {page_size} OFFSET {offset}"
            
            async with db_connection.get_async_connection() as conn:
                if params:
                    rows = await conn.fetch(paginated_query, *params)
                else:
                    rows = await conn.fetch(paginated_query)
            
            if not rows:
                break
            
            # Recordを辞書に変換
            yield [dict(row) for row in rows]
            
            offset += page_size
            
            if len(rows) < page_size:
                break
    
    @staticmethod
    async def bulk_insert(
        table_name: str,
        columns: List[str],
        data: List[tuple],
        batch_size: int = 10000,
        on_conflict: Optional[str] = None
    ) -> int:
        """
        大量データの一括挿入
        """
        total_inserted = 0
        
        # カラムリストを作成
        columns_str = ", ".join(columns)
        placeholders = ", ".join([f"${i+1}" for i in range(len(columns))])
        
        query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
        
        if on_conflict:
            query += f" {on_conflict}"
        
        try:
            # バッチ処理で挿入
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                async with db_connection.get_async_connection() as conn:
                    async with conn.transaction():
                        await conn.executemany(query, batch)
                
                total_inserted += len(batch)
                logger.info(f"Inserted batch {i//batch_size + 1}: {len(batch)} records")
        
        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")
            raise
        
        return total_inserted
    
    @staticmethod
    async def create_index(
        table_name: str,
        index_name: str,
        columns: List[str],
        unique: bool = False,
        concurrent: bool = True,
        where_clause: Optional[str] = None
    ):
        """
        インデックスを作成（大量データ処理の高速化用）
        """
        unique_str = "UNIQUE" if unique else ""
        concurrent_str = "CONCURRENTLY" if concurrent else ""
        columns_str = ", ".join(columns)
        where_str = f"WHERE {where_clause}" if where_clause else ""
        
        query = f"""
        CREATE {unique_str} INDEX {concurrent_str} IF NOT EXISTS {index_name}
        ON {table_name} ({columns_str}) {where_str}
        """
        
        async with db_connection.get_async_connection() as conn:
            await conn.execute(query)
            logger.info(f"Index {index_name} created successfully")
    
    @staticmethod
    async def analyze_table(table_name: str):
        """
        テーブルの統計情報を更新（クエリ最適化用）
        """
        query = f"ANALYZE {table_name}"
        
        async with db_connection.get_async_connection() as conn:
            await conn.execute(query)
            logger.info(f"Table {table_name} analyzed successfully")
    
    @staticmethod
    async def vacuum_table(table_name: str, full: bool = False):
        """
        テーブルのVACUUM実行（ストレージ最適化用）
        """
        vacuum_type = "VACUUM FULL" if full else "VACUUM"
        query = f"{vacuum_type} {table_name}"
        
        # VACUUMはトランザクション外で実行する必要がある
        conn = await db_connection._async_pool.acquire()
        try:
            await conn.execute(query)
            logger.info(f"{vacuum_type} on {table_name} completed successfully")
        finally:
            await db_connection._async_pool.release(conn)
    
    @staticmethod
    async def estimate_row_count(table_name: str) -> int:
        """
        テーブルの推定行数を取得（高速）
        """
        query = """
        SELECT reltuples::BIGINT AS estimate
        FROM pg_class
        WHERE relname = $1
        """
        
        async with db_connection.get_async_connection() as conn:
            result = await conn.fetchval(query, table_name)
            return result or 0
    
    @staticmethod
    async def get_table_size(table_name: str) -> Dict[str, Any]:
        """
        テーブルのサイズ情報を取得
        """
        query = """
        SELECT
            pg_size_pretty(pg_total_relation_size($1)) as total_size,
            pg_size_pretty(pg_relation_size($1)) as table_size,
            pg_size_pretty(pg_indexes_size($1)) as indexes_size
        """
        
        async with db_connection.get_async_connection() as conn:
            row = await conn.fetchrow(query, table_name)
            return dict(row) if row else {}


# シングルトンインスタンス
db_utils = DatabaseUtils()