"""
MySIMOKA Configuration
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # App
    APP_NAME: str = "MySIMOKA"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./mysimoka.db"
    
    # Face Recognition Settings
    EMBEDDING_DIM: int = 128  # Dimensi vektor embedding
    
    # Threshold untuk matching
    THRESHOLD_VERIFIED: float = 0.85   # > 0.85 = Match
    THRESHOLD_UNCERTAIN: float = 0.70  # 0.70 - 0.85 = Uncertain
    # < 0.70 = No Match
    
    class Config:
        env_file = ".env"


settings = Settings()
