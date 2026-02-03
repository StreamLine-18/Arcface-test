"""
MySIMOKA - Real Face Recognition Demo with MediaPipe
Demo webcam untuk test 3-Vector Face Recognition dengan deteksi wajah MediaPipe

MediaPipe Face Landmarker dapat mendeteksi 478 landmark wajah dari berbagai sudut,
termasuk sudut 45 derajat yang sulit untuk Haar Cascade.

Controls:
- SPACE: Capture wajah untuk enrollment/recognition  
- E: Mode Enrollment (daftar wajah baru)
- R: Mode Recognition (kenali wajah)
- D: Delete semua data
- 0-9: Switch camera (index 0-9)
- C: Cycle to next camera
- Q/ESC: Quit
"""

import cv2
import numpy as np
import json
import os
import urllib.request
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# MediaPipe new API (0.10+)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ========================================
# CONFIGURATION
# ========================================

# Threshold untuk matching (raised to reduce false positives)
THRESHOLD_VERIFIED = 0.85   # > 0.85 = Match (was 0.70)
THRESHOLD_UNCERTAIN = 0.70  # 0.70 - 0.85 = Uncertain (was 0.55)

# Face embedding settings
EMBEDDING_DIM = 478 * 3  # 478 landmarks * 3 (x, y, z) = 1434 dimensi

# Database file
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILE = os.path.join(BASE_DIR, "data", "face_database.json")

# Model paths
MODEL_DIR = Path(BASE_DIR) / "data" / "models"
FACE_LANDMARKER_MODEL = MODEL_DIR / "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# ========================================
# MODEL DOWNLOAD
# ========================================

def download_model():
    """Download MediaPipe Face Landmarker model if not exists."""
    MODEL_DIR.mkdir(exist_ok=True)
    
    if not FACE_LANDMARKER_MODEL.exists():
        print(f"📥 Downloading Face Landmarker model...")
        print(f"   From: {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, FACE_LANDMARKER_MODEL)
        print(f"   ✓ Model saved to: {FACE_LANDMARKER_MODEL}")
    else:
        print(f"✓ Model already exists: {FACE_LANDMARKER_MODEL}")


# ========================================
# FACE DATA
# ========================================

@dataclass
class FaceData:
    """Data wajah yang terdeteksi."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    landmarks: np.ndarray            # 478 x 3 (x, y, z)
    confidence: float
    yaw: float   # Rotasi horizontal (kiri-kanan)
    pitch: float # Rotasi vertikal (atas-bawah)


# ========================================
# MEDIAPIPE FACE LANDMARKER
# ========================================

class MediaPipeFaceDetector:
    """
    Deteksi wajah menggunakan MediaPipe Face Landmarker (New API).
    
    Keunggulan:
    - Deteksi 478 landmark wajah dengan akurat
    - Bekerja baik pada sudut hingga 45-60 derajat
    - Dapat mengestimasi pose kepala (yaw, pitch, roll)
    """
    
    # Key landmark indices untuk pose estimation
    NOSE_TIP = 1
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    CHIN = 152
    FOREHEAD = 10
    LEFT_CHEEK = 234
    RIGHT_CHEEK = 454
    
    def __init__(self):
        # Download model if needed
        download_model()
        
        # Create Face Landmarker
        base_options = mp_python.BaseOptions(
            model_asset_path=str(FACE_LANDMARKER_MODEL)
        )
        
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True  # Untuk pose estimation
        )
        
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        print("✓ MediaPipe Face Landmarker initialized")
    
    def detect(self, frame: np.ndarray) -> Optional[FaceData]:
        """
        Deteksi wajah dan ekstrak landmarks.
        
        Args:
            frame: BGR image dari OpenCV
            
        Returns:
            FaceData atau None jika tidak ada wajah
        """
        h, w = frame.shape[:2]
        
        # Convert BGR ke RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Detect
        result = self.landmarker.detect(mp_image)
        
        if not result.face_landmarks:
            return None
        
        # Get first face
        face_lms = result.face_landmarks[0]
        
        # Convert to numpy array (478 landmarks x 3)
        landmarks = np.array([
            [lm.x, lm.y, lm.z]
            for lm in face_lms
        ], dtype=np.float32)
        
        # Calculate bounding box
        x_coords = landmarks[:, 0] * w
        y_coords = landmarks[:, 1] * h
        
        padding = 20
        x_min = int(max(0, x_coords.min() - padding))
        y_min = int(max(0, y_coords.min() - padding))
        x_max = int(min(w, x_coords.max() + padding))
        y_max = int(min(h, y_coords.max() + padding))
        
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        # Estimate pose
        yaw, pitch = self._estimate_pose(landmarks)
        
        # Get confidence from transformation matrix if available
        confidence = 0.9
        
        return FaceData(
            bbox=bbox,
            landmarks=landmarks,
            confidence=confidence,
            yaw=yaw,
            pitch=pitch
        )
    
    def _estimate_pose(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """
        Estimasi pose kepala dari landmarks.
        
        Uses geometric relationship between key landmarks.
        """
        # Get key points
        nose = landmarks[self.NOSE_TIP]
        l_eye = landmarks[self.LEFT_EYE_OUTER]
        r_eye = landmarks[self.RIGHT_EYE_OUTER]
        l_cheek = landmarks[self.LEFT_CHEEK]
        r_cheek = landmarks[self.RIGHT_CHEEK]
        
        # Calculate yaw (horizontal rotation)
        # Based on relative distance of nose to eyes
        nose_to_left = abs(nose[0] - l_eye[0])
        nose_to_right = abs(nose[0] - r_eye[0])
        
        total = nose_to_left + nose_to_right
        if total > 0:
            yaw_ratio = (nose_to_right - nose_to_left) / total
            yaw = yaw_ratio * 60  # Scale to approximately degrees
        else:
            yaw = 0
        
        # Calculate pitch (vertical rotation)
        eye_y = (l_eye[1] + r_eye[1]) / 2
        nose_y = nose[1]
        pitch = (nose_y - eye_y) * 100  # Rough approximation
        
        return yaw, pitch
    
    def draw_landmarks(self, frame: np.ndarray, face_data: FaceData, 
                      draw_all: bool = False) -> np.ndarray:
        """Draw face landmarks on frame."""
        h, w = frame.shape[:2]
        
        # Key landmarks to highlight
        key_landmarks = [
            self.NOSE_TIP, self.LEFT_EYE_OUTER, self.RIGHT_EYE_OUTER,
            self.CHIN, self.LEFT_CHEEK, self.RIGHT_CHEEK
        ]
        
        for i, (x, y, z) in enumerate(face_data.landmarks):
            px = int(x * w)
            py = int(y * h)
            
            if i in key_landmarks:
                # Key landmarks - larger green dots
                cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
            elif draw_all:
                # Other landmarks - tiny gray dots
                cv2.circle(frame, (px, py), 1, (150, 150, 150), -1)
        
        # Draw face contour connections (simplified)
        contour_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                          397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                          172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
        
        points = [(int(face_data.landmarks[i][0] * w), 
                   int(face_data.landmarks[i][1] * h)) for i in contour_indices]
        
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i+1], (0, 255, 255), 1)
        
        return frame


# ========================================
# FACE EMBEDDER (berdasarkan Landmarks)
# ========================================

class LandmarkEmbedder:
    """
    Ekstraksi embedding dari landmark wajah MediaPipe.
    
    Strategi:
    - Gunakan 478 landmarks (x, y, z) sebagai embedding
    - Normalisasi geometri agar invariant terhadap posisi dan skala
    - Total dimensi: 478 * 3 = 1434
    """
    
    def extract(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract normalized embedding dari landmarks.
        
        Normalisasi:
        1. Center menggunakan nose tip sebagai origin
        2. Scale berdasarkan jarak antar mata
        3. Normalize ke unit vector
        """
        # Key indices
        nose_tip = 1
        left_eye = 33
        right_eye = 263
        
        # Get reference points
        nose = landmarks[nose_tip].copy()
        l_eye = landmarks[left_eye]
        r_eye = landmarks[right_eye]
        
        # Calculate eye distance for scaling
        eye_dist = np.linalg.norm(l_eye[:2] - r_eye[:2])
        if eye_dist < 1e-6:
            eye_dist = 1.0
        
        # Center and scale
        centered = landmarks.copy()
        centered -= nose  # Center at nose
        centered /= eye_dist  # Scale by eye distance
        
        # Flatten
        flat = centered.flatten()
        
        # Normalize to unit vector
        norm = np.linalg.norm(flat)
        if norm > 0:
            normalized = flat / norm
        else:
            normalized = flat
        
        return normalized.astype(np.float32)


# ========================================
# FACE MATCHER
# ========================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(vec1, vec2))


def multi_view_match(
    query: np.ndarray, 
    views: Dict[str, np.ndarray]
) -> Tuple[float, str]:
    """Multi-view matching dengan Max Pooling."""
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
    """Simple JSON database untuk menyimpan embeddings with caching."""
    
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self.data = self._load()
        self._users_cache = None  # Cache for converted users
        self._cache_valid = False
    
    def _load(self) -> dict:
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return {"users": []}
    
    def _save(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.data, f)
        self._cache_valid = False  # Invalidate cache
    
    def _rebuild_cache(self):
        """Build numpy cache for all users - called once."""
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
        """Add new user with 3 embeddings."""
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
        """Get all users with cached numpy arrays."""
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
# MAIN DEMO APP
# ========================================

class MySIMOKADemo:
    """Demo app untuk 3-Vector Face Recognition dengan MediaPipe."""
    
    def __init__(self):
        print("\n" + "="*60)
        print("  MySIMOKA - MediaPipe 3-Vector Face Recognition")
        print("="*60)
        
        self.detector = MediaPipeFaceDetector()
        self.embedder = LandmarkEmbedder()
        self.db = FaceDatabase()
        
        # State
        self.mode = "RECOGNITION"
        self.enrollment_step = 0
        self.enrollment_name = ""
        self.enrollment_embeddings = {}
        
        # UI settings
        self.window_name = "MySIMOKA - MediaPipe Face Recognition"
        self.show_landmarks = True
        self.show_all_landmarks = False
        
        # Camera settings
        self.camera_id = 0
        self.cap = None
    
    def get_pose_status(self, yaw: float) -> Tuple[str, Tuple[int, int, int]]:
        """Get pose status text and color."""
        if abs(yaw) < 8:
            return "FRONT", (0, 255, 0)
        elif yaw > 15:
            return "RIGHT", (255, 165, 0)
        elif yaw < -15:
            return "LEFT", (255, 165, 0)
        else:
            return "SLIGHT", (255, 255, 0)
    
    def draw_ui(self, frame: np.ndarray, face_data: Optional[FaceData], match_info: dict):
        """Draw UI overlay."""
        h, w = frame.shape[:2]
        
        # Header
        cv2.rectangle(frame, (0, 0), (w, 65), (25, 25, 25), -1)
        cv2.putText(frame, "MySIMOKA", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, "MediaPipe 478 Landmarks", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        
        # Mode & Stats
        mode_color = (0, 255, 0) if self.mode == "RECOGNITION" else (0, 165, 255)
        cv2.putText(frame, self.mode, (w - 180, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        cv2.putText(frame, f"Users: {self.db.count()} | Cam: {self.camera_id}", (w - 180, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        if face_data:
            x, y, bw, bh = face_data.bbox
            
            # Bounding box
            box_color = (0, 255, 0) if match_info.get("matched") else (0, 200, 255)
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), box_color, 2)
            
            # Landmarks
            if self.show_landmarks:
                frame = self.detector.draw_landmarks(frame, face_data, self.show_all_landmarks)
            
            # Pose info
            pose_text, pose_color = self.get_pose_status(face_data.yaw)
            cv2.putText(frame, f"{pose_text} ({face_data.yaw:+.0f}deg)", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, pose_color, 1)
            
            # Match result
            result = match_info.get("result", "")
            name = match_info.get("name")
            sim = match_info.get('similarity', 0)
            
            panel_y = y + bh + 5
            cv2.rectangle(frame, (x-2, panel_y), (x + 220, panel_y + 55), (0, 0, 0), -1)
            cv2.rectangle(frame, (x-2, panel_y), (x + 220, panel_y + 55), box_color, 2)
            
            # Show name or UNKNOWN
            if name:
                cv2.putText(frame, name, (x + 8, panel_y + 22), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "UNKNOWN PERSON", (x + 8, panel_y + 22), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 255), 2)
            
            # Result colors
            colors = {"VERIFIED": (0, 255, 0), "UNCERTAIN": (0, 165, 255), "UNKNOWN": (0, 0, 255), "NO DATA": (128, 128, 128)}
            result_color = colors.get(result, (200, 200, 200))
            
            cv2.putText(frame, f"{result} {sim:.0%}", (x + 8, panel_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, result_color, 1)
            
            if match_info.get("view") and name:
                cv2.putText(frame, f"[{match_info['view']}]", (x + 150, panel_y + 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Enrollment panel
        if self.mode == "ENROLLMENT":
            views = [("FRONT", "Look straight", (0, 255, 0)),
                    ("RIGHT", "Turn head right ~30°", (255, 165, 0)),
                    ("LEFT", "Turn head left ~30°", (255, 165, 0))]
            
            view_name, instruction, color = views[self.enrollment_step]
            
            cv2.rectangle(frame, (0, h-80), (w, h), (0, 70, 0), -1)
            cv2.putText(frame, f"ENROLLING: {self.enrollment_name}", (15, h-55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Step {self.enrollment_step+1}/3: {view_name} - {instruction}", 
                       (15, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            cv2.putText(frame, "Press SPACE to capture", (15, h-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)
        
        # Help
        cv2.putText(frame, "[E]nroll [R]ecognize [L]andmarks [A]ll [D]elete [0-9/C]Camera [Q]uit [SPACE]Capture", 
                   (10, h-3), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 80, 80), 1)
        
        return frame
    
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
            # Below threshold - treat as UNKNOWN person
            result = "UNKNOWN"
            matched = False
            name = None  # Don't show any name!
        
        return {
            "matched": matched,
            "name": name,
            "similarity": best_sim,
            "result": result,
            "view": best_view
        }
    
    def switch_camera(self, camera_id: int) -> bool:
        """
        Switch to different camera.
        Uses DirectShow backend on Windows for better compatibility with USB cameras.
        """
        print(f"🔄 Switching to camera {camera_id}...")
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Use DirectShow backend on Windows for better USB camera support
        # This helps with Logitech Brio and other high-res cameras
        import platform
        if platform.system() == 'Windows':
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            # Try to reopen previous camera
            if platform.system() == 'Windows':
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.camera_id)
            return False
        
        # Set buffer size to 1 for lower latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set resolution (720p for compatibility, Brio defaults to 4K which is slow)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Set FPS
        self.cap.set(cv2.CAP_PROP_FPS, 24)
        
        # Try to grab a test frame with timeout
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        ret, _ = self.cap.read()
        if not ret:
            print(f"⚠️ Camera {camera_id} opened but cannot read frames")
        
        self.camera_id = camera_id
        
        # Get actual resolution
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"📷 Camera {camera_id} ready: {actual_w}x{actual_h} @ {actual_fps}fps")
        return True
    
    def run(self):
        """Run the demo."""
        # Initialize camera with DirectShow on Windows
        import platform
        if platform.system() == 'Windows':
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.camera_id)
            
        if not self.cap.isOpened():
            print("❌ Cannot open webcam!")
            return
        
        # Camera settings for better compatibility
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 24)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        print("\n📐 478 landmark face detection")
        print("🔄 Supports angles up to 45-60°")
        print(f"📷 Camera: {self.camera_id}")
        print("\nControls:")
        print("  E - Enroll new face")
        print("  R - Recognition mode")
        print("  L - Toggle key landmarks")
        print("  A - Toggle all landmarks")
        print("  D - Delete all data")
        print("  0-9 - Switch camera")
        print("  C - Cycle to next camera")
        print("  SPACE - Capture")
        print("  Q/ESC - Quit")
        print("\n" + "-"*40)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ Failed to grab frame, retrying...")
                continue
            
            frame = cv2.flip(frame, 1)
            face_data = self.detector.detect(frame)
            
            match_info = {"matched": False}
            if face_data and self.mode == "RECOGNITION":
                embedding = self.embedder.extract(face_data.landmarks)
                match_info = self.recognize_face(embedding)
            
            display = self.draw_ui(frame.copy(), face_data, match_info)
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
                print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
            elif key == ord('a'):
                self.show_all_landmarks = not self.show_all_landmarks
                print(f"All landmarks: {'ON' if self.show_all_landmarks else 'OFF'}")
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
                print("🔍 Recognition mode")
            elif key == ord('d'):
                self.db.clear()
                print("🗑️ All data deleted")
            elif key == ord('c'):
                # Cycle to next camera
                next_cam = (self.camera_id + 1) % 10
                self.switch_camera(next_cam)
            elif ord('0') <= key <= ord('9'):
                # Direct camera switch
                cam_id = key - ord('0')
                self.switch_camera(cam_id)
            elif key == ord(' '):
                if not face_data:
                    print("⚠️ No face detected!")
                    continue
                
                embedding = self.embedder.extract(face_data.landmarks)
                
                if self.mode == "ENROLLMENT":
                    views = ["front", "right", "left"]
                    view = views[self.enrollment_step]
                    pose_text, _ = self.get_pose_status(face_data.yaw)
                    
                    self.enrollment_embeddings[view] = embedding
                    print(f"   ✓ {view.upper()} captured (pose: {pose_text}, yaw: {face_data.yaw:+.1f}°)")
                    
                    self.enrollment_step += 1
                    
                    if self.enrollment_step >= 3:
                        user_id = self.db.add_user(
                            self.enrollment_name,
                            self.enrollment_embeddings["front"],
                            self.enrollment_embeddings["right"],
                            self.enrollment_embeddings["left"]
                        )
                        print(f"\n   ✅ {self.enrollment_name} enrolled! (ID: {user_id})")
                        print(f"   Total users: {self.db.count()}")
                        self.mode = "RECOGNITION"
                        self.enrollment_step = 0
                        self.enrollment_embeddings = {}
                    else:
                        steps = ["STRAIGHT", "RIGHT", "LEFT"]
                        print(f"   Step {self.enrollment_step+1}: Turn {steps[self.enrollment_step]}, press SPACE")
                else:
                    result = self.recognize_face(embedding)
                    pose_text, _ = self.get_pose_status(face_data.yaw)
                    print(f"\n🔍 Result ({pose_text}):")
                    print(f"   {result['name'] or 'Unknown'} - {result['similarity']:.1%} [{result['result']}]")
        
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    demo = MySIMOKADemo()
    demo.run()
