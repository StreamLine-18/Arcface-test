"""
Test MySIMOKA Face Recognition against LFW Dataset

This script tests if your enrolled face can be correctly distinguished from 
thousands of celebrity faces in the LFW (Labeled Faces in the Wild) dataset.

Test scenarios:
1. Load your enrolled face embeddings from face_database.json
2. Download LFW dataset using scikit-learn
3. Generate embeddings for LFW faces using MediaPipe
4. Compare your embeddings against all LFW embeddings
5. Verify that your face doesn't match any LFW faces (no false positives)
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import urllib.request

# MediaPipe
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Scikit-learn for LFW dataset
try:
    from sklearn.datasets import fetch_lfw_people
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ scikit-learn not installed. Run: pip install scikit-learn")

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = Path(BASE_DIR) / "data" / "models"
FACE_LANDMARKER_MODEL = MODEL_DIR / "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DB_FILE = os.path.join(BASE_DIR, "data", "face_database.json")

THRESHOLD_VERIFIED = 0.70
THRESHOLD_UNCERTAIN = 0.55


def download_model():
    """Download MediaPipe model if not exists."""
    MODEL_DIR.mkdir(exist_ok=True)
    if not FACE_LANDMARKER_MODEL.exists():
        print(f"📥 Downloading Face Landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, FACE_LANDMARKER_MODEL)
        print(f"   ✓ Model saved")


class LandmarkEmbedder:
    """Extract normalized embedding from landmarks."""
    
    def extract(self, landmarks: np.ndarray) -> np.ndarray:
        nose_tip = 1
        left_eye = 33
        right_eye = 263
        
        nose = landmarks[nose_tip].copy()
        l_eye = landmarks[left_eye]
        r_eye = landmarks[right_eye]
        
        eye_dist = np.linalg.norm(l_eye[:2] - r_eye[:2])
        if eye_dist < 1e-6:
            eye_dist = 1.0
        
        centered = landmarks.copy()
        centered -= nose
        centered /= eye_dist
        
        flat = centered.flatten()
        norm = np.linalg.norm(flat)
        if norm > 0:
            normalized = flat / norm
        else:
            normalized = flat
        
        return normalized.astype(np.float32)


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


def test_against_lfw():
    """Main test function."""
    print("\n" + "="*60)
    print("  MySIMOKA vs LFW Dataset Test")
    print("="*60)
    
    if not HAS_SKLEARN:
        print("❌ Please install scikit-learn first:")
        print("   pip install scikit-learn")
        return
    
    # Load enrolled users
    enrolled_users = load_enrolled_users()
    if not enrolled_users:
        print("❌ No enrolled users found!")
        print("   Please run demo_webcam.py and enroll your face first.")
        return
    
    print(f"\n✓ Found {len(enrolled_users)} enrolled users:")
    for u in enrolled_users:
        print(f"   - {u['name']} (ID: {u['id']})")
    
    # Download MediaPipe model
    download_model()
    
    # Initialize MediaPipe Face Landmarker
    print("\n📦 Initializing MediaPipe...")
    base_options = mp_python.BaseOptions(
        model_asset_path=str(FACE_LANDMARKER_MODEL)
    )
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    embedder = LandmarkEmbedder()
    
    # Download LFW dataset
    print("\n📥 Downloading LFW dataset (this may take a while)...")
    print("   Dataset: Labeled Faces in the Wild")
    print("   Source: http://vis-www.cs.umass.edu/lfw/")
    
    lfw = fetch_lfw_people(
        min_faces_per_person=5,  # Only people with 5+ images
        resize=1.0,
        color=True,
        download_if_missing=True
    )
    
    n_samples = len(lfw.images)
    n_people = len(lfw.target_names)
    print(f"\n✓ LFW Dataset loaded:")
    print(f"   - {n_samples} images")
    print(f"   - {n_people} unique people")
    
    # Process LFW images and generate embeddings
    print("\n🔄 Processing LFW faces...")
    lfw_embeddings = []
    lfw_names = []
    processed = 0
    failed = 0
    
    # Process a subset for faster testing
    max_samples = min(500, n_samples)  # Limit to 500 for speed
    print(f"   Processing first {max_samples} images...")
    
    for i in range(max_samples):
        img = lfw.images[i]
        name = lfw.target_names[lfw.target[i]]
        
        # Convert to uint8 RGB (LFW images are 0-255 float)
        img_uint8 = img.astype(np.uint8)
        
        # Convert RGB to BGR for consistency
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        
        # Convert back to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Detect face
        result = landmarker.detect(mp_image)
        
        if result.face_landmarks:
            landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]
            ], dtype=np.float32)
            
            embedding = embedder.extract(landmarks)
            lfw_embeddings.append(embedding)
            lfw_names.append(name)
            processed += 1
        else:
            failed += 1
        
        if (i + 1) % 50 == 0:
            print(f"   Processed {i+1}/{max_samples} images...")
    
    print(f"\n✓ LFW processing complete:")
    print(f"   - Successfully processed: {processed}")
    print(f"   - Failed (no face detected): {failed}")
    
    if not lfw_embeddings:
        print("❌ No LFW embeddings generated!")
        return
    
    # Test each enrolled user against LFW
    print("\n" + "="*60)
    print("  Testing Enrolled Users vs LFW Faces")
    print("="*60)
    
    for user in enrolled_users:
        print(f"\n👤 Testing: {user['name']}")
        print("-" * 40)
        
        false_positives = []
        max_similarity = -1
        max_sim_name = ""
        
        for view_name, user_embedding in user['embeddings'].items():
            for i, lfw_emb in enumerate(lfw_embeddings):
                sim = cosine_similarity(user_embedding, lfw_emb)
                
                if sim > max_similarity:
                    max_similarity = sim
                    max_sim_name = lfw_names[i]
                
                if sim > THRESHOLD_VERIFIED:
                    false_positives.append({
                        "lfw_name": lfw_names[i],
                        "similarity": sim,
                        "view": view_name
                    })
        
        print(f"   Max similarity with LFW: {max_similarity:.2%}")
        print(f"   Most similar LFW person: {max_sim_name}")
        print(f"   Threshold (VERIFIED): {THRESHOLD_VERIFIED:.2%}")
        
        if false_positives:
            print(f"\n   ⚠️ FALSE POSITIVES: {len(false_positives)}")
            for fp in false_positives[:5]:  # Show top 5
                print(f"      - {fp['lfw_name']}: {fp['similarity']:.2%} [{fp['view']}]")
        else:
            print(f"\n   ✅ NO FALSE POSITIVES!")
            print(f"      Your face is correctly distinguished from {processed} LFW faces")
        
        # Calculate margin
        margin = THRESHOLD_VERIFIED - max_similarity
        if margin > 0:
            print(f"   📊 Safety margin: {margin:.2%} below threshold")
        else:
            print(f"   ⚠️ Above threshold by: {-margin:.2%}")
    
    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    print(f"   - Enrolled users tested: {len(enrolled_users)}")
    print(f"   - LFW faces compared: {processed}")
    print(f"   - Threshold: {THRESHOLD_VERIFIED:.2%}")
    
    landmarker.close()
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_against_lfw()
