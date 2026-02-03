"""
Database Configuration
AsyncIO SQLAlchemy setup untuk SQLite
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from app.config import settings
from app.models.face_embedding import Base

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True
)

# Create async session factory
async_session = async_sessionmaker(
    engine, 
    class_=AsyncSession,
    expire_on_commit=False
)


async def init_db():
    """Initialize database - create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database initialized!")


async def get_db():
    """Dependency untuk mendapatkan database session."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
