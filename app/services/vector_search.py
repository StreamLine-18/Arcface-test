"""
Vector Search Service
Implementasi Multi-View Face Matching untuk identifikasi user
"""

import numpy as np
from typing import Optional, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import Session
from dataclasses import dataclass

from app.models.face_embedding import FaceEmbedding
from app.utils.vector_utils import (
    cosine_similarity,
    multi_view_match,
    classify_match_result,
    MatchResult,
    MatchScore
)
from app.config import settings


@dataclass
class SearchResult:
    """Hasil pencarian wajah"""
    user_id: Optional[int]
    user_name: Optional[str]
    similarity: float
    result: str  # "verified", "uncertain", "no_match"
    matched_view: Optional[str]
    all_scores: List[dict]  # Detail similarity untuk semua user


class VectorSearchService:
    """
    Service untuk mencari dan mencocokkan wajah dengan Multi-View strategy.
    
    Strategi:
    1. Query embedding dibandingkan dengan SEMUA user di database
    2. Untuk setiap user, hitung similarity dengan 3 view (front, right, left)
    3. Ambil MAX similarity dari 3 view tersebut (Max Pooling)
    4. User dengan similarity tertinggi adalah kandidat match
    5. Klasifikasi berdasarkan threshold
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.threshold_verified = settings.THRESHOLD_VERIFIED
        self.threshold_uncertain = settings.THRESHOLD_UNCERTAIN
    
    async def search(self, query_embedding: np.ndarray) -> SearchResult:
        """
        Cari user yang cocok dengan query embedding.
        
        Args:
            query_embedding: Vektor wajah yang akan dicocokkan
            
        Returns:
            SearchResult dengan detail matching
        """
        # Get semua users dari database
        result = await self.db.execute(select(FaceEmbedding))
        all_users = result.scalars().all()
        
        if not all_users:
            return SearchResult(
                user_id=None,
                user_name=None,
                similarity=0.0,
                result=MatchResult.NO_MATCH.value,
                matched_view=None,
                all_scores=[]
            )
        
        # Hitung similarity untuk setiap user
        all_scores = []
        best_match = None
        best_similarity = -1.0
        best_view = None
        
        for user in all_users:
            # Get 3 embeddings untuk user ini
            embeddings = user.get_embeddings()
            
            # Multi-view matching
            similarity, matched_view = multi_view_match(query_embedding, embeddings)
            
            # Simpan detail untuk debugging/analysis
            user_score = {
                "user_id": user.id,
                "user_name": user.user_name,
                "similarity": round(similarity, 4),
                "matched_view": matched_view,
                "details": {
                    "front": round(cosine_similarity(query_embedding, embeddings["front"]), 4),
                    "right": round(cosine_similarity(query_embedding, embeddings["right"]), 4),
                    "left": round(cosine_similarity(query_embedding, embeddings["left"]), 4)
                }
            }
            all_scores.append(user_score)
            
            # Track best match
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = user
                best_view = matched_view
        
        # Klasifikasi hasil
        match_result = classify_match_result(
            best_similarity,
            self.threshold_verified,
            self.threshold_uncertain
        )
        
        return SearchResult(
            user_id=best_match.id if best_match else None,
            user_name=best_match.user_name if best_match else None,
            similarity=round(best_similarity, 4),
            result=match_result.value,
            matched_view=best_view,
            all_scores=sorted(all_scores, key=lambda x: x["similarity"], reverse=True)
        )
    
    async def verify_user(
        self, 
        user_id: int, 
        query_embedding: np.ndarray
    ) -> Tuple[bool, float, str]:
        """
        Verifikasi apakah wajah cocok dengan user tertentu (1:1 matching).
        
        Args:
            user_id: ID user yang akan diverifikasi
            query_embedding: Vektor wajah
            
        Returns:
            Tuple (is_verified, similarity, matched_view)
        """
        result = await self.db.execute(
            select(FaceEmbedding).where(FaceEmbedding.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return False, 0.0, None
        
        embeddings = user.get_embeddings()
        similarity, matched_view = multi_view_match(query_embedding, embeddings)
        
        is_verified = similarity > self.threshold_verified
        
        return is_verified, round(similarity, 4), matched_view
    
    async def compare_two_embeddings(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> dict:
        """
        Bandingkan dua embedding secara langsung.
        Berguna untuk testing similarity antar sudut.
        
        Args:
            embedding1: Vektor pertama
            embedding2: Vektor kedua
            
        Returns:
            Dict dengan similarity dan klasifikasi
        """
        similarity = cosine_similarity(embedding1, embedding2)
        result = classify_match_result(
            similarity,
            self.threshold_verified,
            self.threshold_uncertain
        )
        
        return {
            "similarity": round(similarity, 4),
            "result": result.value,
            "thresholds": {
                "verified": self.threshold_verified,
                "uncertain": self.threshold_uncertain
            }
        }
