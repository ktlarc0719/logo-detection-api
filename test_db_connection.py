"""
データベース接続テストスクリプト
"""

import asyncio
from src.db.connection import db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def test_database_connection():
    """データベース接続をテスト"""
    try:
        # 接続テスト
        logger.info("Testing database connection...")
        is_connected = await db_connection.test_connection()
        
        if is_connected:
            logger.info("✓ Database connection successful!")
            
            # テーブル一覧を取得
            async with db_connection.get_async_connection() as conn:
                tables = await conn.fetch("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)
                
                logger.info(f"\nAvailable tables ({len(tables)}):")
                for table in tables:
                    logger.info(f"  - {table['table_name']}")
                
                # データベース情報を取得
                db_info = await conn.fetchrow("""
                    SELECT 
                        current_database() as database,
                        current_user as user,
                        version() as version
                """)
                
                logger.info(f"\nDatabase info:")
                logger.info(f"  Database: {db_info['database']}")
                logger.info(f"  User: {db_info['user']}")
                logger.info(f"  Version: {db_info['version'].split(',')[0]}")
        
        else:
            logger.error("✗ Database connection failed!")
            
    except Exception as e:
        logger.error(f"Error during database test: {e}")
        raise
    
    finally:
        # 接続プールを閉じる
        await db_connection.close_async()


if __name__ == "__main__":
    asyncio.run(test_database_connection())