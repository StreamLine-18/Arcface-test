"""
Vector Utilities untuk Face Recognition
Implementasi Cosine Similarity dan Multi-View Matching
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum


class MatchResult(Enum):
    """Hasil pencocokan wajah"""
    VERIFIED = "verified"      # Cocok dengan confidence tinggi
    UNCERTAIN = "uncertain"    # Mungkin cocok, perlu verifikasi
    NO_MATCH = "no_match"      # Tidak cocok


@dataclass
class MatchScore:
    """Detail hasil matching"""
    user_id: Optional[int]
    user_name: Optional[str]
    similarity: float
    result: MatchResult
    matched_view: Optional[str]  # "front", "right", atau "left"


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalisasi vektor ke unit vector (panjang = 1).
    Diperlukan untuk cosine similarity yang akurat.
    
    Args:
        vector: Vektor embedding
        
    Returns:
        Unit vector
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Hitung Cosine Similarity antara dua vektor.
    
    Untuk vektor yang sudah dinormalisasi:
    cosine_similarity = dot_product(vec1, vec2)
    
    Range: -1 (berlawanan) sampai 1 (identik)
    
    Args:
        vec1: Vektor pertama
        vec2: Vektor kedua
        
    Returns:
        Nilai similarity [-1, 1]
    """
    # Pastikan vektor sudah dinormalisasi
    vec1_norm = normalize_vector(vec1)
    vec2_norm = normalize_vector(vec2)
    
    return float(np.dot(vec1_norm, vec2_norm))


def generate_mock_embedding(
    dim: int = 128, 
    base_vector: Optional[np.ndarray] = None,
    noise_level: float = 0.1
) -> np.ndarray:
    """
    Generate mock embedding untuk testing.
    
    Jika base_vector diberikan, akan menghasilkan vektor yang mirip
    (mensimulasikan sudut wajah berbeda dari orang yang sama).
    
    Args:
        dim: Dimensi embedding
        base_vector: Vektor dasar untuk variasi
        noise_level: Tingkat noise (0-1), semakin tinggi semakin berbeda
        
    Returns:
        Mock embedding vector (normalized)
    """
    if base_vector is not None:
        # Buat variasi dari base vector (simulasi sudut berbeda)
        noise = np.random.randn(dim) * noise_level
        vector = base_vector + noise
    else:
        # Generate vektor random baru
        vector = np.random.randn(dim)
    
    return normalize_vector(vector).astype(np.float32)


def multi_view_match(
    query_embedding: np.ndarray,
    reference_views: Dict[str, np.ndarray]
) -> Tuple[float, str]:
    """
    Multi-View Matching dengan Max Pooling Strategy.
    
    Membandingkan query dengan 3 view reference (depan, kanan, kiri)
    dan mengambil similarity tertinggi.
    
    Args:
        query_embedding: Embedding wajah yang akan dicocokkan
        reference_views: Dict dengan keys "front", "right", "left"
        
    Returns:
        Tuple (max_similarity, best_matching_view)
    """
    best_similarity = -1.0
    best_view = None
    
    for view_name, ref_embedding in reference_views.items():
        similarity = cosine_similarity(query_embedding, ref_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_view = view_name
    
    return best_similarity, best_view


def classify_match_result(
    similarity: float,
    threshold_verified: float = 0.85,
    threshold_uncertain: float = 0.70
) -> MatchResult:
    """
    Klasifikasi hasil matching berdasarkan threshold.
    
    Args:
        similarity: Nilai cosine similarity
        threshold_verified: Threshold untuk VERIFIED
        threshold_uncertain: Threshold untuk UNCERTAIN
        
    Returns:
        MatchResult enum
    """
    if similarity > threshold_verified:
        return MatchResult.VERIFIED
    elif similarity > threshold_uncertain:
        return MatchResult.UNCERTAIN
    else:
        return MatchResult.NO_MATCH


# === Quick Test ===
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Vector Utils")
    print("=" * 50)
    
    # Test 1: Same vector = similarity 1.0
    vec1 = generate_mock_embedding(128)
    sim = cosine_similarity(vec1, vec1)
    print(f"\n1. Same vector similarity: {sim:.4f} (expected: 1.0)")
    
    # Test 2: Variasi kecil = similarity tinggi (orang sama, sudut beda)
    vec_front = generate_mock_embedding(128)
    vec_right = generate_mock_embedding(128, base_vector=vec_front, noise_level=0.1)
    vec_left = generate_mock_embedding(128, base_vector=vec_front, noise_level=0.1)
    
    print(f"\n2. Simulasi orang yang sama dari sudut berbeda:")
    print(f"   Front vs Right: {cosine_similarity(vec_front, vec_right):.4f}")
    print(f"   Front vs Left:  {cosine_similarity(vec_front, vec_left):.4f}")
    print(f"   Right vs Left:  {cosine_similarity(vec_right, vec_left):.4f}")
    
    # Test 3: Vektor random = similarity rendah (orang berbeda)
    vec_stranger = generate_mock_embedding(128)
    print(f"\n3. Orang berbeda:")
    print(f"   Known vs Stranger: {cosine_similarity(vec_front, vec_stranger):.4f}")
    
    # Test 4: Multi-view matching
    print(f"\n4. Multi-View Matching Test:")
    query = generate_mock_embedding(128, base_vector=vec_right, noise_level=0.05)
    reference_views = {
        "front": vec_front,
        "right": vec_right,
        "left": vec_left
    }
    
    best_sim, best_view = multi_view_match(query, reference_views)
    result = classify_match_result(best_sim)
    
    print(f"   Best match: {best_view} with similarity {best_sim:.4f}")
    print(f"   Result: {result.value}")
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)
