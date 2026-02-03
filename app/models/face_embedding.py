"""
Face Embedding Model
SQLAlchemy model untuk menyimpan vektor wajah user
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, LargeBinary, JSON
from sqlalchemy.orm import declarative_base
from datetime import datetime
import numpy as np
import json

Base = declarative_base()


class FaceEmbedding(Base):
    """
    Model untuk menyimpan 3-vektor wajah per user.
    
    Setiap user memiliki 3 embedding:
    - front: Wajah menghadap depan (0°)
    - right: Wajah serong kanan (~30°)
    - left: Wajah serong kiri (~30°)
    """
    
    __tablename__ = "face_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_name = Column(String(100), nullable=False, index=True)
    
    # 3 vektor wajah disimpan sebagai JSON string
    embedding_front = Column(String, nullable=False)   # Vektor depan
    embedding_right = Column(String, nullable=False)   # Vektor kanan
    embedding_left = Column(String, nullable=False)    # Vektor kiri
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def set_embeddings(self, front: np.ndarray, right: np.ndarray, left: np.ndarray):
        """
        Set ketiga embedding dari numpy arrays.
        
        Args:
            front: Embedding wajah depan
            right: Embedding wajah kanan
            left: Embedding wajah kiri
        """
        self.embedding_front = json.dumps(front.tolist())
        self.embedding_right = json.dumps(right.tolist())
        self.embedding_left = json.dumps(left.tolist())
    
    def get_embeddings(self) -> dict:
        """
        Get semua embedding sebagai dict of numpy arrays.
        
        Returns:
            Dict dengan keys "front", "right", "left"
        """
        return {
            "front": np.array(json.loads(self.embedding_front), dtype=np.float32),
            "right": np.array(json.loads(self.embedding_right), dtype=np.float32),
            "left": np.array(json.loads(self.embedding_left), dtype=np.float32)
        }
    
    def to_dict(self) -> dict:
        """Convert model ke dictionary untuk API response."""
        return {
            "id": self.id,
            "user_name": self.user_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "has_embeddings": all([
                self.embedding_front,
                self.embedding_right,
                self.embedding_left
            ])
        }
