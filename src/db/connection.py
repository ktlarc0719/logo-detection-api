"""
PostgreSQL データベース接続管理
"""

import asyncio
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
import asyncpg
from asyncpg import Pool, Connection
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor

from src.db.config import db_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseConnection:
    """データベース接続管理クラス"""
    
    def __init__(self):
        self._sync_pool: Optional[SimpleConnectionPool] = None
        self._async_pool: Optional[Pool] = None
        self._is_initialized = False
    
    async def init_async(self):
        """非同期接続プールの初期化"""
        if not self._async_pool:
            try:
                self._async_pool = await asyncpg.create_pool(
                    host=db_config.db_host,
                    port=db_config.db_port,
                    user=db_config.db_user,
                    password=db_config.db_password,
                    database=db_config.db_name,
                    min_size=db_config.db_pool_min_size,
                    max_size=db_config.db_pool_max_size,
                    timeout=db_config.db_pool_timeout,
                    command_timeout=db_config.db_command_timeout
                )
                logger.info("Async database pool created successfully")
            except Exception as e:
                logger.error(f"Failed to create async database pool: {e}")
                raise
    
    def init_sync(self):
        """同期接続プールの初期化"""
        if not self._sync_pool:
            try:
                self._sync_pool = SimpleConnectionPool(
                    db_config.db_pool_min_size,
                    db_config.db_pool_max_size,
                    host=db_config.db_host,
                    port=db_config.db_port,
                    user=db_config.db_user,
                    password=db_config.db_password,
                    database=db_config.db_name
                )
                logger.info("Sync database pool created successfully")
            except Exception as e:
                logger.error(f"Failed to create sync database pool: {e}")
                raise
    
    @asynccontextmanager
    async def get_async_connection(self) -> AsyncGenerator[Connection, None]:
        """非同期データベース接続を取得"""
        if not self._async_pool:
            await self.init_async()
        
        async with self._async_pool.acquire() as connection:
            yield connection
    
    def get_sync_connection(self):
        """同期データベース接続を取得"""
        if not self._sync_pool:
            self.init_sync()
        
        return self._sync_pool.getconn()
    
    def put_sync_connection(self, connection):
        """同期データベース接続を返却"""
        if self._sync_pool:
            self._sync_pool.putconn(connection)
    
    async def close_async(self):
        """非同期接続プールを閉じる"""
        if self._async_pool:
            await self._async_pool.close()
            self._async_pool = None
            logger.info("Async database pool closed")
    
    def close_sync(self):
        """同期接続プールを閉じる"""
        if self._sync_pool:
            self._sync_pool.closeall()
            self._sync_pool = None
            logger.info("Sync database pool closed")
    
    async def test_connection(self) -> bool:
        """データベース接続をテスト"""
        try:
            async with self.get_async_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                logger.info("Database connection test successful")
                return result == 1
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    async def execute_query(self, query: str, *args, timeout: Optional[float] = None):
        """クエリを実行"""
        async with self.get_async_connection() as conn:
            return await conn.fetch(query, *args, timeout=timeout)
    
    async def execute_many(self, query: str, args_list: list, batch_size: Optional[int] = None):
        """バッチクエリを実行（大量データ処理用）"""
        batch_size = batch_size or db_config.batch_size
        async with self.get_async_connection() as conn:
            # トランザクション内でバッチ処理
            async with conn.transaction():
                for i in range(0, len(args_list), batch_size):
                    batch = args_list[i:i + batch_size]
                    await conn.executemany(query, batch)
                    logger.debug(f"Processed batch {i//batch_size + 1}")


# シングルトンインスタンス
db_connection = DatabaseConnection()


# 便利な関数
async def get_db() -> AsyncGenerator[Connection, None]:
    """FastAPI依存性注入用のデータベース接続取得関数"""
    async with db_connection.get_async_connection() as conn:
        yield conn