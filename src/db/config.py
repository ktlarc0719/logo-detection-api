"""
PostgreSQL データベース接続設定
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """データベース設定"""
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "logo_detection")
    db_user: str = os.getenv("DB_USER", "admin")
    db_password: str = os.getenv("DB_PASSWORD", "Inagaki1")
    
    # 接続プール設定（大量データ処理用）
    db_pool_min_size: int = int(os.getenv("DB_POOL_MIN_SIZE", "5"))
    db_pool_max_size: int = int(os.getenv("DB_POOL_MAX_SIZE", "20"))
    db_pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    db_command_timeout: int = int(os.getenv("DB_COMMAND_TIMEOUT", "60"))
    
    # バッチ処理設定
    batch_size: int = int(os.getenv("DB_BATCH_SIZE", "1000"))
    
    @property
    def database_url(self) -> str:
        """データベース接続URL"""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    @property
    def async_database_url(self) -> str:
        """非同期データベース接続URL"""
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # 余分な環境変数を無視
    }


# シングルトンインスタンス
db_config = DatabaseConfig()