"""
MySIMOKA - Face Recognition Demo with InsightFace/ArcFace

This version uses deep learning (ArcFace) for proper face recognition.
ArcFace produces 512-dimensional embeddings that capture IDENTITY, not just geometry.

Controls:
  E - Enroll new face (3-step: front, right, left)
  R - Switch to Recognition mode
  L - Toggle landmarks
  D - Delete all data
  0-9 - Switch camera
  C - Cycle to next camera
  SPACE - Capture
  Q/ESC - Quit
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, NamedTuple

# InsightFace for deep learning face recognition
import insightface
from insightface.app import FaceAnalysis

# ========================================
# CONFIGURATION
# ========================================

# Threshold untuk matching - ArcFace uses different scale
# ArcFace similarity is typically 0.0 to 1.0, but can be higher
THRESHOLD_VERIFIED = 0.45   # > 0.45 = Verified (ArcFace typical)
THRESHOLD_UNCERTAIN = 0.30  # 0.30 - 0.45 = Uncertain

# Database file
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILE = os.path.join(BASE_DIR, "data", "face_database_arcface.json")

# ========================================
# ARCFACE EMBEDDER
# ========================================

class ArcFaceRecognizer:
    """Face recognition using InsightFace/ArcFace - Speed Optimized."""
    
    def __init__(self):
        print("📦 Loading ArcFace model (buffalo_s - balanced)...")
        # Use buffalo_s - balanced speed/accuracy
        self.app = FaceAnalysis(
            name='buffalo_s',  # Medium model
            providers=['CPUExecutionProvider']
        )
        # Detection size
        self.app.prepare(ctx_id=0, det_size=(320, 320))
        print("✓ ArcFace model loaded!")
    
    def detect_and_embed(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, dict]]:
        """
        Detect face and extract 512-dim embedding.
        Returns (embedding, face_info) or None if no face.
        """
        # InsightFace expects BGR
        faces = self.app.get(frame)
        
        if not faces:
            return None
        
        # Get largest face
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        
        # Extract info
        bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
        embedding = face.normed_embedding  # 512-dim normalized
        
        # Landmarks (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
        landmarks = face.kps if hasattr(face, 'kps') else None
        
        face_info = {
            'bbox': bbox,
            'landmarks': landmarks,
            'det_score': float(face.det_score) if hasattr(face, 'det_score') else 0
        }
        
        return embedding, face_info
    
    def get_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Get just the embedding."""
        result = self.detect_and_embed(frame)
        if result:
            return result[0]
        return None


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(vec1, vec2))


def multi_view_match(
    query: np.ndarray, 
    views: Dict[str, np.ndarray]
) -> Tuple[float, str]:
    """Multi-view matching with Max Pooling."""
    best_sim = -1.0
    best_view = None
    
    for view_name, embedding in views.items():
        sim = cosine_similarity(query, embedding)
        if sim > best_sim:
            best_sim = sim
            best_view = view_name
    
    return best_sim, best_view


# ========================================
# DATABASE
# ========================================

class FaceDatabase:
    """Database for ArcFace embeddings."""
    
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self.data = self._load()
        self._users_cache = None
        self._cache_valid = False
    
    def _load(self) -> dict:
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return {"users": []}
    
    def _save(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.data, f)
        self._cache_valid = False
    
    def _rebuild_cache(self):
        self._users_cache = []
        for u in self.data["users"]:
            user = {
                "id": u["id"],
                "name": u["name"],
                "embeddings": {
                    k: np.array(v, dtype=np.float32) 
                    for k, v in u["embeddings"].items()
                }
            }
            self._users_cache.append(user)
        self._cache_valid = True
        print(f"   📦 Cached {len(self._users_cache)} users")
    
    def add_user(self, name: str, front: np.ndarray, right: np.ndarray, left: np.ndarray) -> int:
        user_id = len(self.data["users"]) + 1
        
        user = {
            "id": user_id,
            "name": name,
            "embeddings": {
                "front": front.tolist(),
                "right": right.tolist(),
                "left": left.tolist()
            },
            "created_at": datetime.now().isoformat()
        }
        
        self.data["users"].append(user)
        self._save()
        return user_id
    
    def get_all_users(self) -> List[dict]:
        if not self._cache_valid:
            self._rebuild_cache()
        return self._users_cache
    
    def clear(self):
        self.data = {"users": []}
        self._save()
        self._users_cache = []
        self._cache_valid = True
    
    def count(self) -> int:
        return len(self.data["users"])


# ========================================
# MAIN DEMO
# ========================================

class ArcFaceDemo:
    """Demo with ArcFace recognition - Optimized."""
    
    def __init__(self):
        print("\n" + "="*60)
        print("  MySIMOKA - ArcFace Deep Learning Recognition")
        print("="*60)
        
        self.recognizer = ArcFaceRecognizer()
        self.db = FaceDatabase()
        
        # State
        self.mode = "RECOGNITION"
        self.enrollment_step = 0
        self.enrollment_name = ""
        self.enrollment_embeddings = {}
        
        # UI
        self.window_name = "MySIMOKA - ArcFace Recognition"
        self.show_landmarks = True
        
        # Camera
        self.camera_id = 0
        self.cap = None
        
        # Performance optimization
        self.frame_count = 0
        self.process_every_n = 5  # Process every 5th frame
        self.cached_face_info = None
        self.cached_embedding = None
        self.cached_match_info = {"matched": False}
        self.fps_display = 0
    
    def recognize_face(self, embedding: np.ndarray) -> dict:
        """Recognize face against database."""
        users = self.db.get_all_users()
        
        if not users:
            return {"matched": False, "name": None, "similarity": 0, "result": "NO DATA"}
        
        best_match = None
        best_sim = -1
        best_view = None
        
        for user in users:
            sim, view = multi_view_match(embedding, user["embeddings"])
            if sim > best_sim:
                best_sim = sim
                best_match = user
                best_view = view
        
        if best_sim > THRESHOLD_VERIFIED:
            result = "VERIFIED"
            matched = True
            name = best_match["name"]
        elif best_sim > THRESHOLD_UNCERTAIN:
            result = "UNCERTAIN"
            matched = False
            name = best_match["name"] + " ?"
        else:
            result = "UNKNOWN"
            matched = False
            name = None
        
        return {
            "matched": matched,
            "name": name,
            "similarity": best_sim,
            "result": result,
            "view": best_view
        }
    
    def draw_ui(self, frame: np.ndarray, face_info: Optional[dict], match_info: dict) -> np.ndarray:
        """Draw UI overlay."""
        h, w = frame.shape[:2]
        
        # Header
        cv2.rectangle(frame, (0, 0), (w, 65), (30, 30, 30), -1)
        cv2.putText(frame, "MySIMOKA ArcFace", (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Deep Learning Face Recognition", (15, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        # Mode indicator
        mode_color = (0, 255, 0) if self.mode == "RECOGNITION" else (0, 165, 255)
        cv2.putText(frame, self.mode, (w - 180, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        cv2.putText(frame, f"Users: {self.db.count()} | Skip: {self.process_every_n} | FPS: {self.fps_display:.0f}", 
                   (w - 280, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        if face_info:
            x1, y1, x2, y2 = face_info['bbox']
            
            # Bounding box
            box_color = (0, 255, 0) if match_info.get("matched") else (0, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Landmarks (5 points)
            if self.show_landmarks and face_info.get('landmarks') is not None:
                for pt in face_info['landmarks']:
                    cv2.circle(frame, tuple(pt.astype(int)), 3, (0, 255, 255), -1)
            
            # Detection score
            det_score = face_info.get('det_score', 0)
            cv2.putText(frame, f"Det: {det_score:.0%}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Match result panel
            result = match_info.get("result", "")
            name = match_info.get("name")
            sim = match_info.get('similarity', 0)
            
            panel_y = y2 + 5
            cv2.rectangle(frame, (x1-2, panel_y), (x1 + 220, panel_y + 55), (0, 0, 0), -1)
            cv2.rectangle(frame, (x1-2, panel_y), (x1 + 220, panel_y + 55), box_color, 2)
            
            if name:
                cv2.putText(frame, name, (x1 + 8, panel_y + 22), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "UNKNOWN PERSON", (x1 + 8, panel_y + 22), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 255), 2)
            
            colors = {"VERIFIED": (0, 255, 0), "UNCERTAIN": (0, 165, 255), "UNKNOWN": (0, 0, 255)}
            result_color = colors.get(result, (200, 200, 200))
            
            cv2.putText(frame, f"{result} {sim:.0%}", (x1 + 8, panel_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, result_color, 1)
        
        # Enrollment panel
        if self.mode == "ENROLLMENT":
            views = [("FRONT", "Look straight"), ("RIGHT", "Turn right ~30°"), ("LEFT", "Turn left ~30°")]
            view_name, instruction = views[self.enrollment_step]
            
            cv2.rectangle(frame, (0, h-80), (w, h), (0, 70, 0), -1)
            cv2.putText(frame, f"ENROLLING: {self.enrollment_name}", (15, h-55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Step {self.enrollment_step+1}/3: {view_name} - {instruction}", 
                       (15, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            cv2.putText(frame, "Press SPACE to capture", (15, h-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)
        
        # Help
        cv2.putText(frame, "[E]nroll [R]ecognize [L]andmarks [D]elete [0-9/C]Camera [Q]uit [SPACE]Capture", 
                   (10, h-3), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 80, 80), 1)
        
        return frame
    
    def switch_camera(self, camera_id: int) -> bool:
        print(f"🔄 Switching to camera {camera_id}...")
        
        if self.cap is not None:
            self.cap.release()
        
        import platform
        if platform.system() == 'Windows':
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return False
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 24)
        
        self.camera_id = camera_id
        print(f"📷 Camera {camera_id} ready")
        return True
    
    def run(self):
        """Main loop with frame skipping optimization."""
        if not self.switch_camera(0):
            return
        
        print("\nControls:")
        print("  E - Enroll | R - Recognize | L - Landmarks")
        print("  D - Delete | 0-9/C - Camera | Q - Quit")
        print("  +/- - Adjust skip frames (faster/slower)")
        print("-" * 40)
        
        import time
        last_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            self.frame_count += 1
            
            # Only process every Nth frame for performance
            if self.frame_count % self.process_every_n == 0:
                result = self.recognizer.detect_and_embed(frame)
                
                if result:
                    self.cached_embedding, self.cached_face_info = result
                    if self.mode == "RECOGNITION":
                        self.cached_match_info = self.recognize_face(self.cached_embedding)
                else:
                    self.cached_face_info = None
                    self.cached_embedding = None
                    self.cached_match_info = {"matched": False}
                
                # Calculate FPS
                current_time = time.time()
                self.fps_display = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
                last_time = current_time
            
            # Use cached results for display
            display = self.draw_ui(frame.copy(), self.cached_face_info, self.cached_match_info)
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
            elif key == ord('+') or key == ord('='):
                self.process_every_n = max(1, self.process_every_n - 1)
                print(f"⚡ Process every {self.process_every_n} frames (faster)")
            elif key == ord('-'):
                self.process_every_n = min(15, self.process_every_n + 1)
                print(f"🐌 Process every {self.process_every_n} frames (slower)")
            elif key == ord('e'):
                name = input("\n👤 Enter name: ").strip()
                if name:
                    self.mode = "ENROLLMENT"
                    self.enrollment_name = name
                    self.enrollment_step = 0
                    self.enrollment_embeddings = {}
                    print(f"📝 Enrolling: {name}")
                    print("   Step 1: Look STRAIGHT, press SPACE")
            elif key == ord('r'):
                self.mode = "RECOGNITION"
            elif key == ord('d'):
                self.db.clear()
                print("🗑️ Database cleared")
            elif key == ord('c'):
                self.switch_camera((self.camera_id + 1) % 10)
            elif ord('0') <= key <= ord('9'):
                self.switch_camera(key - ord('0'))
            elif key == ord(' '):
                # Force process current frame for capture
                result = self.recognizer.detect_and_embed(frame)
                if result:
                    embedding, _ = result
                else:
                    embedding = None
                
                if embedding is None:
                    print("⚠️ No face detected!")
                    continue
                
                if self.mode == "ENROLLMENT":
                    views = ["front", "right", "left"]
                    view = views[self.enrollment_step]
                    
                    self.enrollment_embeddings[view] = embedding
                    print(f"   ✓ {view.upper()} captured")
                    
                    self.enrollment_step += 1
                    
                    if self.enrollment_step >= 3:
                        user_id = self.db.add_user(
                            self.enrollment_name,
                            self.enrollment_embeddings["front"],
                            self.enrollment_embeddings["right"],
                            self.enrollment_embeddings["left"]
                        )
                        print(f"\n   ✅ {self.enrollment_name} enrolled! (ID: {user_id})")
                        self.mode = "RECOGNITION"
                        self.enrollment_step = 0
                    else:
                        steps = ["STRAIGHT", "RIGHT", "LEFT"]
                        print(f"   Step {self.enrollment_step+1}: Turn {steps[self.enrollment_step]}, press SPACE")
                else:
                    result = self.recognize_face(embedding)
                    print(f"\n🔍 {result['name'] or 'Unknown'} - {result['similarity']:.1%} [{result['result']}]")
        
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    demo = ArcFaceDemo()
    demo.run()
