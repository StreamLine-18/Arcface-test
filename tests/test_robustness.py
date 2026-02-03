"""
Test MySIMOKA Face Recognition Robustness

This script tests the face recognition system's ability to correctly 
distinguish enrolled faces from random/fake embeddings.

Since LFW download failed, we'll use:
1. Random synthetic embeddings (simulating unknown faces)
2. Noisy versions of your own embeddings (simulating same person with variations)
"""

import numpy as np
import json
import os
from typing import List, Dict

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILE = os.path.join(BASE_DIR, "data", "face_database.json")
THRESHOLD_VERIFIED = 0.70
THRESHOLD_UNCERTAIN = 0.55
EMBEDDING_DIM = 478 * 3  # 1434 dimensions


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity."""
    return float(np.dot(vec1, vec2))


def load_enrolled_users() -> List[Dict]:
    """Load enrolled users from database."""
    if not os.path.exists(DB_FILE):
        return []
    
    with open(DB_FILE, 'r') as f:
        data = json.load(f)
    
    users = []
    for u in data.get("users", []):
        user = {
            "id": u["id"],
            "name": u["name"],
            "embeddings": {
                k: np.array(v, dtype=np.float32) 
                for k, v in u["embeddings"].items()
            }
        }
        users.append(user)
    return users


def generate_random_embeddings(n: int) -> np.ndarray:
    """Generate random unit-normalized embeddings (simulating unknown faces)."""
    embeddings = np.random.randn(n, EMBEDDING_DIM).astype(np.float32)
    # Normalize each embedding to unit vector
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


def generate_noisy_embedding(original: np.ndarray, noise_level: float) -> np.ndarray:
    """Add noise to an embedding (simulating same person with variations)."""
    noise = np.random.randn(*original.shape).astype(np.float32) * noise_level
    noisy = original + noise
    # Renormalize
    norm = np.linalg.norm(noisy)
    if norm > 0:
        noisy = noisy / norm
    return noisy


def run_tests():
    """Run all robustness tests."""
    print("\n" + "="*60)
    print("  MySIMOKA Face Recognition Robustness Test")
    print("="*60)
    
    # Load enrolled users
    enrolled_users = load_enrolled_users()
    if not enrolled_users:
        print("❌ No enrolled users found!")
        print("   Please run demo_webcam.py and enroll your face first.")
        return
    
    print(f"\n✓ Found {len(enrolled_users)} enrolled users:")
    for u in enrolled_users:
        print(f"   - {u['name']} (ID: {u['id']})")
    
    # ========================================
    # TEST 1: Random Unknown Faces
    # ========================================
    print("\n" + "-"*60)
    print("TEST 1: Random Unknown Faces (False Positive Test)")
    print("-"*60)
    print("   Generating 1000 random embeddings (simulating unknown faces)...")
    
    random_embeddings = generate_random_embeddings(1000)
    
    for user in enrolled_users:
        print(f"\n👤 Testing: {user['name']}")
        
        false_positives = 0
        max_sim = -1
        similarities = []
        
        for user_emb in user['embeddings'].values():
            for rand_emb in random_embeddings:
                sim = cosine_similarity(user_emb, rand_emb)
                similarities.append(sim)
                if sim > max_sim:
                    max_sim = sim
                if sim > THRESHOLD_VERIFIED:
                    false_positives += 1
        
        avg_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        print(f"   Statistics against 1000 random embeddings:")
        print(f"   - Max similarity: {max_sim:.2%}")
        print(f"   - Avg similarity: {avg_sim:.2%}")
        print(f"   - Std deviation:  {std_sim:.2%}")
        print(f"   - Threshold:      {THRESHOLD_VERIFIED:.2%}")
        
        if false_positives == 0:
            print(f"   ✅ PASS: No false positives!")
            margin = THRESHOLD_VERIFIED - max_sim
            print(f"   📊 Safety margin: {margin:.2%} below threshold")
        else:
            print(f"   ❌ FAIL: {false_positives} false positives detected!")
    
    # ========================================
    # TEST 2: Self-Recognition with Noise
    # ========================================
    print("\n" + "-"*60)
    print("TEST 2: Self-Recognition with Noise (True Positive Test)")
    print("-"*60)
    print("   Testing if we can still recognize you with noise...")
    
    noise_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    for user in enrolled_users:
        print(f"\n👤 Testing: {user['name']}")
        
        for noise_level in noise_levels:
            correct = 0
            total = 0
            
            for view_name, user_emb in user['embeddings'].items():
                # Generate 10 noisy versions of this embedding
                for _ in range(10):
                    noisy_emb = generate_noisy_embedding(user_emb, noise_level)
                    
                    # Check against all stored embeddings
                    max_sim = -1
                    for stored_emb in user['embeddings'].values():
                        sim = cosine_similarity(noisy_emb, stored_emb)
                        if sim > max_sim:
                            max_sim = sim
                    
                    total += 1
                    if max_sim > THRESHOLD_VERIFIED:
                        correct += 1
            
            accuracy = (correct / total) * 100 if total > 0 else 0
            status = "✅" if accuracy >= 90 else "⚠️" if accuracy >= 70 else "❌"
            print(f"   Noise {noise_level:.0%}: {accuracy:.0f}% recognition rate {status}")
    
    # ========================================
    # TEST 3: Cross-User Confusion
    # ========================================
    if len(enrolled_users) > 1:
        print("\n" + "-"*60)
        print("TEST 3: Cross-User Confusion (Discrimination Test)")
        print("-"*60)
        print("   Testing if different users are correctly distinguished...")
        
        for i, user1 in enumerate(enrolled_users):
            for j, user2 in enumerate(enrolled_users):
                if i >= j:
                    continue
                
                max_sim = -1
                for emb1 in user1['embeddings'].values():
                    for emb2 in user2['embeddings'].values():
                        sim = cosine_similarity(emb1, emb2)
                        if sim > max_sim:
                            max_sim = sim
                
                status = "✅" if max_sim < THRESHOLD_VERIFIED else "❌"
                print(f"   {user1['name']} vs {user2['name']}: {max_sim:.2%} {status}")
    
    # ========================================
    # TEST 4: Embedding Quality Check
    # ========================================
    print("\n" + "-"*60)
    print("TEST 4: Embedding Quality Check")
    print("-"*60)
    
    for user in enrolled_users:
        print(f"\n👤 User: {user['name']}")
        
        # Check view consistency
        views = list(user['embeddings'].keys())
        for i, v1 in enumerate(views):
            for j, v2 in enumerate(views):
                if i >= j:
                    continue
                sim = cosine_similarity(
                    user['embeddings'][v1], 
                    user['embeddings'][v2]
                )
                print(f"   {v1} ↔ {v2}: {sim:.2%}")
        
        # Check embedding norms
        for view, emb in user['embeddings'].items():
            norm = np.linalg.norm(emb)
            print(f"   {view} norm: {norm:.4f} (should be ~1.0)")
    
    # Summary
    print("\n" + "="*60)
    print("  Test Summary")
    print("="*60)
    print(f"   ✅ Random face rejection: TESTED")
    print(f"   ✅ Noise tolerance: TESTED")
    if len(enrolled_users) > 1:
        print(f"   ✅ User discrimination: TESTED")
    print(f"   ✅ Embedding quality: TESTED")
    print("\n   🎉 All tests complete!")


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    run_tests()
