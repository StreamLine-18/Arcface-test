"""
MySIMOKA - Mock App untuk Testing 3-Vector Face Recognition

Aplikasi ini mensimulasikan sistem pengenalan wajah dengan pendekatan
Multi-View (3 sudut: depan, kanan, kiri) untuk menguji efektivitas
sebelum deployment ke Raspberry Pi.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional, List
from pydantic import BaseModel
import numpy as np

from app.config import settings
from app.database import init_db, get_db
from app.services.auth import AuthService
from app.services.vector_search import VectorSearchService
from app.utils.vector_utils import generate_mock_embedding, cosine_similarity


# === Pydantic Models ===

class UserRegisterRequest(BaseModel):
    """Request untuk registrasi user"""
    user_name: str


class UserResponse(BaseModel):
    """Response data user"""
    id: int
    user_name: str
    created_at: Optional[str]
    has_embeddings: bool


class SearchRequest(BaseModel):
    """Request untuk search - bisa pakai mock atau embedding custom"""
    # Untuk testing dengan mock
    mock_from_user_id: Optional[int] = None  # Generate query mirip user ini
    mock_noise: float = 0.1  # Noise level untuk simulasi sudut berbeda
    
    # Atau custom embedding (list of floats)
    custom_embedding: Optional[List[float]] = None


class SearchResponse(BaseModel):
    """Response hasil pencarian"""
    matched_user_id: Optional[int]
    matched_user_name: Optional[str]
    similarity: float
    result: str
    matched_view: Optional[str]
    all_scores: List[dict]


class VerifyRequest(BaseModel):
    """Request untuk verifikasi 1:1"""
    user_id: int
    mock_noise: float = 0.1


class VerifyResponse(BaseModel):
    """Response verifikasi"""
    is_verified: bool
    similarity: float
    matched_view: Optional[str]


class CompareRequest(BaseModel):
    """Request untuk membandingkan 2 embedding"""
    user1_id: int
    user2_id: int
    view1: str = "front"  # front, right, left
    view2: str = "front"


class SimulationRequest(BaseModel):
    """Request untuk simulasi multi-user"""
    num_users: int = 5
    test_samples_per_user: int = 3
    noise_levels: List[float] = [0.05, 0.10, 0.15, 0.20]


# === Lifespan ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup dan shutdown events."""
    # Startup
    await init_db()
    print(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} started!")
    print(f"📊 Thresholds: VERIFIED > {settings.THRESHOLD_VERIFIED}, UNCERTAIN > {settings.THRESHOLD_UNCERTAIN}")
    yield
    # Shutdown
    print("👋 Shutting down...")


# === App Instance ===

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## MySIMOKA Mock App
    
    Aplikasi untuk menguji efektivitas **3-Vector Face Recognition**:
    - **Front view**: Wajah menghadap depan (0°)
    - **Right view**: Wajah serong kanan (~30°)  
    - **Left view**: Wajah serong kiri (~30°)
    
    ### Strategi Matching
    - Multi-View Cosine Similarity
    - Max Pooling (ambil similarity tertinggi dari 3 view)
    - Threshold-based classification
    
    ### Endpoints
    - `/register` - Daftarkan user dengan mock embeddings
    - `/search` - Cari user yang cocok dengan query
    - `/verify/{user_id}` - Verifikasi 1:1 dengan user tertentu
    - `/simulate` - Jalankan simulasi multi-user
    """,
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Endpoints ===

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "thresholds": {
            "verified": settings.THRESHOLD_VERIFIED,
            "uncertain": settings.THRESHOLD_UNCERTAIN
        }
    }


@app.post("/register", response_model=UserResponse)
async def register_user(request: UserRegisterRequest, db=Depends(get_db)):
    """
    Daftarkan user baru dengan mock embeddings.
    
    Akan menghasilkan 3 vektor wajah (front, right, left) secara otomatis
    untuk simulasi pendaftaran.
    """
    auth_service = AuthService(db)
    
    # Check if user exists
    existing = await auth_service.get_user_by_name(request.user_name)
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")
    
    user = await auth_service.register_user_mock(request.user_name)
    
    return UserResponse(
        id=user.id,
        user_name=user.user_name,
        created_at=user.created_at.isoformat() if user.created_at else None,
        has_embeddings=True
    )


@app.get("/users", response_model=List[UserResponse])
async def list_users(db=Depends(get_db)):
    """Get semua users terdaftar."""
    auth_service = AuthService(db)
    users = await auth_service.get_all_users()
    
    return [
        UserResponse(
            id=u.id,
            user_name=u.user_name,
            created_at=u.created_at.isoformat() if u.created_at else None,
            has_embeddings=True
        )
        for u in users
    ]


@app.delete("/users/{user_id}")
async def delete_user(user_id: int, db=Depends(get_db)):
    """Hapus user by ID."""
    auth_service = AuthService(db)
    deleted = await auth_service.delete_user(user_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": f"User {user_id} deleted"}


@app.post("/search", response_model=SearchResponse)
async def search_face(request: SearchRequest, db=Depends(get_db)):
    """
    Cari user yang cocok dengan query embedding.
    
    Modes:
    1. `mock_from_user_id`: Generate query mirip user tertentu (untuk testing)
    2. `custom_embedding`: Gunakan embedding custom
    
    Tanpa keduanya akan generate random embedding (unknown person).
    """
    auth_service = AuthService(db)
    search_service = VectorSearchService(db)
    
    # Determine query embedding
    if request.custom_embedding:
        query = np.array(request.custom_embedding, dtype=np.float32)
    elif request.mock_from_user_id:
        # Get user's embedding and add noise
        user = await auth_service.get_user_by_id(request.mock_from_user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        embeddings = user.get_embeddings()
        base_embedding = embeddings["front"]  # Use front as base
        query = generate_mock_embedding(
            128, 
            base_vector=base_embedding, 
            noise_level=request.mock_noise
        )
    else:
        # Random unknown person
        query = generate_mock_embedding(128)
    
    # Search
    result = await search_service.search(query)
    
    return SearchResponse(
        matched_user_id=result.user_id,
        matched_user_name=result.user_name,
        similarity=result.similarity,
        result=result.result,
        matched_view=result.matched_view,
        all_scores=result.all_scores
    )


@app.post("/verify/{user_id}", response_model=VerifyResponse)
async def verify_face(user_id: int, request: VerifyRequest, db=Depends(get_db)):
    """
    Verifikasi 1:1 - cek apakah wajah cocok dengan user tertentu.
    
    Menggunakan mock embedding dari user yang sama dengan noise
    untuk mensimulasikan wajah dari sudut berbeda.
    """
    auth_service = AuthService(db)
    search_service = VectorSearchService(db)
    
    # Get user
    user = await auth_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Generate query dari embedding user dengan noise
    embeddings = user.get_embeddings()
    query = generate_mock_embedding(
        128,
        base_vector=embeddings["front"],
        noise_level=request.mock_noise
    )
    
    # Verify
    is_verified, similarity, matched_view = await search_service.verify_user(
        user_id, query
    )
    
    return VerifyResponse(
        is_verified=is_verified,
        similarity=similarity,
        matched_view=matched_view
    )


@app.post("/compare")
async def compare_embeddings(request: CompareRequest, db=Depends(get_db)):
    """
    Bandingkan embedding dari 2 user berbeda.
    
    Berguna untuk melihat seberapa berbeda embedding antar orang.
    """
    auth_service = AuthService(db)
    search_service = VectorSearchService(db)
    
    user1 = await auth_service.get_user_by_id(request.user1_id)
    user2 = await auth_service.get_user_by_id(request.user2_id)
    
    if not user1 or not user2:
        raise HTTPException(status_code=404, detail="User not found")
    
    emb1 = user1.get_embeddings()[request.view1]
    emb2 = user2.get_embeddings()[request.view2]
    
    result = await search_service.compare_two_embeddings(emb1, emb2)
    
    return {
        "user1": {"id": user1.id, "name": user1.user_name, "view": request.view1},
        "user2": {"id": user2.id, "name": user2.user_name, "view": request.view2},
        **result
    }


@app.post("/simulate")
async def run_simulation(request: SimulationRequest, db=Depends(get_db)):
    """
    Jalankan simulasi lengkap untuk menguji efektivitas 3-vektor.
    
    1. Daftarkan N users dengan mock embeddings
    2. Untuk setiap user, generate test samples dengan berbagai noise level
    3. Hitung accuracy pada setiap noise level
    
    Returns detail statistik dan rekomendasi threshold.
    """
    auth_service = AuthService(db)
    search_service = VectorSearchService(db)
    
    results = {
        "users_created": [],
        "test_results": [],
        "accuracy_by_noise": {},
        "false_positives": [],
        "recommendations": {}
    }
    
    # 1. Create users
    for i in range(request.num_users):
        user_name = f"SimUser_{i+1}"
        
        # Delete if exists
        existing = await auth_service.get_user_by_name(user_name)
        if existing:
            await auth_service.delete_user(existing.id)
        
        user = await auth_service.register_user_mock(user_name)
        results["users_created"].append({
            "id": user.id,
            "name": user.user_name
        })
    
    # 2. Test each user
    all_users = await auth_service.get_all_users()
    
    for noise in request.noise_levels:
        correct = 0
        total = 0
        
        for user in all_users:
            embeddings = user.get_embeddings()
            
            for _ in range(request.test_samples_per_user):
                # Generate query dengan noise
                query = generate_mock_embedding(
                    128,
                    base_vector=embeddings["front"],
                    noise_level=noise
                )
                
                # Search
                search_result = await search_service.search(query)
                
                is_correct = search_result.user_id == user.id
                if is_correct:
                    correct += 1
                total += 1
                
                results["test_results"].append({
                    "true_user": user.user_name,
                    "predicted_user": search_result.user_name,
                    "noise": noise,
                    "similarity": search_result.similarity,
                    "result": search_result.result,
                    "correct": is_correct
                })
        
        accuracy = correct / total if total > 0 else 0
        results["accuracy_by_noise"][str(noise)] = {
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": total
        }
    
    # 3. Test false positives (unknown person)
    for _ in range(request.num_users):
        unknown_query = generate_mock_embedding(128)
        search_result = await search_service.search(unknown_query)
        
        results["false_positives"].append({
            "matched_user": search_result.user_name,
            "similarity": search_result.similarity,
            "result": search_result.result,
            "is_false_positive": search_result.result == "verified"
        })
    
    # 4. Recommendations
    fp_rate = sum(1 for fp in results["false_positives"] if fp["is_false_positive"]) / len(results["false_positives"])
    
    results["recommendations"] = {
        "false_positive_rate": round(fp_rate, 4),
        "best_noise_level": min(
            results["accuracy_by_noise"].items(),
            key=lambda x: -x[1]["accuracy"]
        )[0],
        "suggestion": (
            "3-vector approach is effective!" 
            if results["accuracy_by_noise"][str(request.noise_levels[1])]["accuracy"] > 0.9 
            else "Consider adjusting thresholds or noise levels"
        )
    }
    
    return results


@app.get("/stats")
async def get_stats(db=Depends(get_db)):
    """Get statistik database."""
    auth_service = AuthService(db)
    users = await auth_service.get_all_users()
    
    return {
        "total_users": len(users),
        "thresholds": {
            "verified": settings.THRESHOLD_VERIFIED,
            "uncertain": settings.THRESHOLD_UNCERTAIN
        },
        "embedding_dim": settings.EMBEDDING_DIM
    }


# === Run ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
