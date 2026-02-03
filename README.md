Face Recognition Demo

Real-time face recognition dengan dua opsi:
- **ArcFace (Recommended)** - Deep learning, akurat untuk identity
- **MediaPipe** - Landmark-based, lebih cepat tapi kurang akurat

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run ArcFace demo (RECOMMENDED - accurate identity recognition)
python demo_arcface.py

# Run MediaPipe demo (faster but less accurate for identity)
python demo_webcam.py
```

---

## ⚠️ Important: Which Demo to Use?

| Feature | demo_arcface.py ✅ | demo_webcam.py |
|---------|-------------------|----------------|
| **Accuracy** | High (identity) | Low (geometry only) |
| **False Positives** | Very low | High ⚠️ |
| **Speed** | ~2-5 FPS detection | ~15-24 FPS |
| **Model** | ArcFace 512-dim | MediaPipe landmarks |
| **Use Case** | Production | Demo/testing |

**Recommendation:** Use `demo_arcface.py` for real face recognition!

## 📷 Webcam Demo Controls

| Key | Action |
|-----|--------|
| **E** | Enroll new face (3-step: front, right, left) |
| **R** | Switch to Recognition mode |
| **L** | Toggle key landmarks display |
| **A** | Toggle all 478 landmarks display |
| **D** | Delete all enrolled data |
| **SPACE** | Capture face / Confirm action |
| **0-9** | Switch to camera 0-9 |
| **C** | Cycle to next camera |
| **Q / ESC** | Quit application |

---

## 🗃️ Database Seeding

### Quick Commands

```bash
# Add 10 fake celebrity-named faces
python manage_db.py setup

# Add more fake faces
python manage_db.py add5      # Add 5 random fake faces
python manage_db.py add20     # Add 20 random fake faces  
python manage_db.py add50     # Add 50 random fake faces

# List all users
python manage_db.py list

# Clear only fake users (keep real enrolled faces)
python manage_db.py clear-fake

# Clear entire database
python manage_db.py clear-all

# Interactive menu
python manage_db.py
```

### Database Structure

Database disimpan di `face_database.json`:

```json
{
  "users": [
    {
      "id": 1,
      "name": "John_Doe",
      "embeddings": {
        "front": [1434 float values],
        "right": [1434 float values],
        "left": [1434 float values]
      },
      "created_at": "2026-02-02T23:30:00",
      "is_fake": false
    }
  ]
}
```

---

## 🧪 Testing

### Robustness Test
```bash
python test_robustness.py
```
Tests:
- Random face rejection (false positive test)
- Noise tolerance
- Cross-user discrimination
- Embedding quality check

### LFW Dataset Test (requires internet)
```bash
python test_against_lfw.py
```
Tests against Labeled Faces in the Wild dataset.

---

## 🔧 Configuration

### Thresholds (in `demo_webcam.py`)

| Threshold | Value | Meaning |
|-----------|-------|---------|
| `THRESHOLD_VERIFIED` | 0.70 | ≥70% = VERIFIED ✅ |
| `THRESHOLD_UNCERTAIN` | 0.55 | 55-70% = UNCERTAIN ⚠️ |
| Below 0.55 | - | NO MATCH ❌ |

### Camera Settings

- Resolution: 1280x720 (720p)
- FPS: 24
- Buffer Size: 1 (low latency)
- Backend: DirectShow (Windows) / Default (Linux/Mac)

---

## 📐 Technical Details

### Face Detection
- **Model**: MediaPipe Face Landmarker (face_landmarker.task)
- **Landmarks**: 478 points per face
- **Angle Support**: Up to 45-60° yaw

### Embedding
- **Dimension**: 1434 floats (478 landmarks × 3 coordinates)
- **Normalization**: Unit vector (L2 norm = 1.0)
- **Centering**: At nose tip
- **Scaling**: By inter-eye distance

### Matching
- **Algorithm**: Cosine Similarity
- **Strategy**: Max Pooling across 3 views (front, right, left)
- **Formula**: `similarity = dot(embedding1, embedding2)`

---

## 📁 Project Structure

```
MySimoka/
├── demo_webcam.py      # Main webcam demo
├── manage_db.py        # Database seeding tool
├── test_robustness.py  # Offline robustness tests
├── test_against_lfw.py # LFW dataset test
├── face_database.json  # User embeddings storage
├── models/
│   └── face_landmarker.task  # MediaPipe model
└── requirements.txt    # Python dependencies
```

---

## 🎯 Enrollment Process

1. Press **E** to start enrollment
2. Enter your name
3. **Step 1**: Look **STRAIGHT** at camera → Press SPACE
4. **Step 2**: Turn head **RIGHT** (~30-45°) → Press SPACE
5. **Step 3**: Turn head **LEFT** (~30-45°) → Press SPACE
6. ✅ Enrollment complete!

---

## ⚡ Performance Tips

- With 200+ users, first frame may be slow (caching embeddings)
- After caching, recognition runs at full framerate
- Use `clear-fake` to remove test data and keep real users
- Logitech Brio cameras work best on camera index 3

---

## 📊 Test Results

| Test | Result |
|------|--------|
| False positive rate (vs 1000 random) | 0% |
| Max similarity vs random faces | ~9% |
| Safety margin below threshold | 61% |
| Self-recognition accuracy | 98-99% |
