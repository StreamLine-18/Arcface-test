"""
Auth Service
Menangani pendaftaran user dengan 3-vektor wajah
"""

import numpy as np
from typing import Optional, Dict, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.face_embedding import FaceEmbedding
from app.utils.vector_utils import generate_mock_embedding, normalize_vector


class AuthService:
    """
    Service untuk mengelola pendaftaran dan data user.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    async def register_user(
        self,
        user_name: str,
        embedding_front: np.ndarray,
        embedding_right: np.ndarray,
        embedding_left: np.ndarray
    ) -> FaceEmbedding:
        """
        Daftarkan user baru dengan 3 vektor wajah.
        
        Args:
            user_name: Nama user
            embedding_front: Vektor wajah depan
            embedding_right: Vektor wajah kanan
            embedding_left: Vektor wajah kiri
            
        Returns:
            FaceEmbedding object yang baru dibuat
        """
        # Normalisasi semua vektor
        front = normalize_vector(embedding_front)
        right = normalize_vector(embedding_right)
        left = normalize_vector(embedding_left)
        
        # Buat record baru
        user = FaceEmbedding(user_name=user_name)
        user.set_embeddings(front, right, left)
        
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        
        return user
    
    async def register_user_mock(self, user_name: str) -> FaceEmbedding:
        """
        Daftarkan user dengan mock embeddings untuk testing.
        
        Menghasilkan 3 vektor yang mirip (simulasi orang yang sama
        dari sudut berbeda) dengan noise level realistis.
        
        Args:
            user_name: Nama user
            
        Returns:
            FaceEmbedding object
        """
        # Generate base embedding (wajah depan)
        base = generate_mock_embedding(128)
        
        # Generate variasi untuk kanan dan kiri
        # Noise level 0.15 mensimulasikan perbedaan sudut ~30°
        right = generate_mock_embedding(128, base_vector=base, noise_level=0.15)
        left = generate_mock_embedding(128, base_vector=base, noise_level=0.15)
        
        return await self.register_user(user_name, base, right, left)
    
    async def get_user_by_id(self, user_id: int) -> Optional[FaceEmbedding]:
        """Get user by ID."""
        result = await self.db.execute(
            select(FaceEmbedding).where(FaceEmbedding.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def get_user_by_name(self, user_name: str) -> Optional[FaceEmbedding]:
        """Get user by name."""
        result = await self.db.execute(
            select(FaceEmbedding).where(FaceEmbedding.user_name == user_name)
        )
        return result.scalar_one_or_none()
    
    async def get_all_users(self) -> List[FaceEmbedding]:
        """Get semua users."""
        result = await self.db.execute(select(FaceEmbedding))
        return result.scalars().all()
    
    async def delete_user(self, user_id: int) -> bool:
        """Hapus user by ID."""
        user = await self.get_user_by_id(user_id)
        if user:
            await self.db.delete(user)
            await self.db.commit()
            return True
        return False
