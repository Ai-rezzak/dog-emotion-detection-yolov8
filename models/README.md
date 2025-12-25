# 🤖 Pre-trained Models

This directory contains the trained YOLOv8 models for dog detection and emotion analysis.

---

## 📦 Available Models

### 1. Dog Detection Model (`dog_detect.pt`)

**Purpose**: Detects dogs in images/videos and filters out other animals

**Specifications**:
- **Architecture**: YOLOv8n (nano)
- **Input Size**: 640x640
- **Output**: Bounding box + confidence score
- **Classes**: 1 (Dog)
- **File Size**: ~6 MB
- **Accuracy**: 89.4%

**Performance**:
| Metric | Value |
|--------|-------|
| Precision | 91.2% |
| Recall | 88.6% |
| F1-Score | 89.9% |
| mAP@0.5 | 92.3% |
| Inference Speed | 30+ FPS (GPU) |

**Use Case**: First-stage detection to ensure emotion analysis only runs on dogs

---

### 2. Emotion Detection Model (`emotion_detect.pt`)

**Purpose**: Analyzes emotional states of detected dogs

**Specifications**:
- **Architecture**: YOLOv8n (nano)
- **Input Size**: 640x640
- **Output**: Emotion class + confidence score
- **Classes**: 4 (Happy, Angry, Sad, Sleeping)
- **File Size**: ~6 MB
- **Overall Accuracy**: 92.3%

**Performance by Class**:
| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| **Happy** 😊 | 97.7% | 96.5% | 97.1% |
| **Angry** 😠 | 86.2% | 84.8% | 85.5% |
| **Sad** 😢 | 97.0% | 95.3% | 96.1% |
| **Sleeping** 😴 | 88.1% | 87.4% | 87.7% |

**Use Case**: Second-stage analysis for emotion classification

---

## 📥 Download Models

Due to GitHub's file size limitations (100MB max), pre-trained models are hosted externally.

### Download Options:

#### Option 1: Google Drive (Recommended)
🔗 [Download Both Models (~12 MB total)](https://drive.google.com/your-link-here)

#### Option 2: Individual Downloads
- 🐕 [dog_detect.pt](https://drive.google.com/your-link) (~6 MB)
- 😊 [emotion_detect.pt](https://drive.google.com/your-link) (~6 MB)

### Installation Steps:

```bash
# 1. Download models from the link above
# 2. Place them in the models/ directory
models/
├── dog_detect.pt
├── emotion_detect.pt
└── README.md
```

---

## 🚀 Quick Start

### Load Models in Python

```python
from ultralytics import YOLO

# Load dog detection model
dog_model = YOLO("models/dog_detect.pt")

# Load emotion detection model
emotion_model = YOLO("models/emotion_detect.pt")
```

### Basic Inference

```python
import cv2

# Load image
image = cv2.imread("test_image.jpg")

# Step 1: Detect dog
dog_results = dog_model(image)

# Step 2: Analyze emotion (if dog detected)
emotion_results = emotion_model(image)

# Display results
dog_results[0].show()
emotion_results[0].show()
```

---

## 🔧 Model Training Details

### Dog Detection Model

**Training Configuration**:
```yaml
Model: YOLOv8n
Dataset: 14,065 dog images
Epochs: 150
Batch Size: 16
Image Size: 640x640
Optimizer: Adam
Learning Rate: 0.01 (with cosine decay)
Augmentation: Mosaic, Mixup, HSV, Flip
Platform: Google Colab (Tesla T4 GPU)
Training Time: ~5 hours
```

**Data Split**:
- Training: 80% (11,252 images)
- Validation: 10% (1,406 images)
- Test: 10% (1,407 images)

---

### Emotion Detection Model

**Training Configuration**:
```yaml
Model: YOLOv8n
Dataset: 4,806 labeled images
Epochs: 100
Batch Size: 16
Image Size: 640x640
Optimizer: Adam
Learning Rate: 0.01 (with cosine decay)
Augmentation: Mosaic, Mixup, HSV, Flip, Rotation
Platform: Google Colab (Tesla T4 GPU)
Training Time: ~3 hours
```

**Data Split**:
- Training: 80% (3,845 images)
- Validation: 10% (480 images)
- Test: 10% (481 images)

**Class Distribution** (Training Set):
- Happy: 1,234 images
- Angry: 730 images
- Sad: 866 images
- Sleeping: 1,015 images

---

## 📊 Model Comparison

| Feature | Dog Detection | Emotion Detection |
|---------|---------------|-------------------|
| **Primary Task** | Object Detection | Classification |
| **Classes** | 1 (Dog) | 4 (Emotions) |
| **Training Samples** | 14,065 | 4,806 |
| **Accuracy** | 89.4% | 92.3% |
| **Inference Speed** | 35 FPS | 40 FPS |
| **Model Size** | 6 MB | 6 MB |
| **Use Case** | Filtering | Analysis |

---

## 🎯 Model Architecture

Both models use **YOLOv8n** (nano variant):

```
Input (640x640x3)
    ↓
Backbone (CSPDarknet)
    ↓
Neck (PANet)
    ↓
Head (Detection/Classification)
    ↓
Output (Bounding Boxes + Classes + Confidence)
```

**Why YOLOv8n?**
- ⚡ Fast inference (real-time capable)
- 🎯 High accuracy
- 📱 Small model size (mobile-friendly)
- 🔄 Single-stage detection (efficient)

---

## 🔄 Cascaded Detection Pipeline

The two models work together in a cascaded pipeline:

```
Input Image/Video
       ↓
┌─────────────────────┐
│ Dog Detection Model │
└─────────────────────┘
       ↓
   Is Dog Detected?
       ↓
    No ─────→ Skip (ignore other animals)
       ↓ Yes
┌─────────────────────┐
│ Emotion Detection   │
│ Model               │
└─────────────────────┘
       ↓
Output: Dog + Emotion Label
```

**Benefits**:
- Prevents false emotion detection on cats, rabbits, etc.
- Improves overall system accuracy
- Reduces computational waste

---

## 💾 Model Files

### dog_detect.pt
- **Format**: PyTorch (.pt)
- **Framework**: Ultralytics YOLOv8
- **Size**: ~6 MB
- **Input**: RGB image (640x640)
- **Output**: [x, y, w, h, confidence, class]

### emotion_detect.pt
- **Format**: PyTorch (.pt)
- **Framework**: Ultralytics YOLOv8
- **Size**: ~6 MB
- **Input**: RGB image (640x640)
- **Output**: [x, y, w, h, confidence, emotion_class]

**Class Mapping**:
```python
{
    0: "Happy",
    1: "Angry",
    2: "Sad",
    3: "Sleeping"
}
```

---

## 🧪 Model Validation

### Validation Methodology

Models were validated using:
- **Hold-out Test Set**: 10% of data never seen during training
- **Cross-validation**: 5-fold for robustness
- **Real-world Testing**: Videos and images from unseen sources
- **Edge Cases**: Different breeds, lighting, angles

### Test Results

**Dog Detection**:
- ✅ Correctly identified 89.4% of dogs
- ✅ Low false positive rate (8.8%)
- ✅ Works across various breeds and sizes

**Emotion Detection**:
- ✅ Overall 92.3% accuracy
- ✅ Best performance: Happy (97.7%) and Sad (97.0%)
- ⚠️ Needs improvement: Angry (86.2%) - more data needed

---

## ⚠️ Known Limitations

### Dog Detection Model:
- May struggle with occluded dogs (partially hidden)
- Lower accuracy on very small dogs (<50px)
- Mixed breeds sometimes misclassified

### Emotion Detection Model:
- "Angry" class has lower accuracy (86.2%)
- Challenges with similar expressions (sad vs sleeping)
- Lighting conditions affect performance
- Works best with frontal/side views

---

## 🔄 Model Updates

### Version History

**v1.0** (Current - 2023)
- Initial release
- 89.4% dog detection accuracy
- 92.3% emotion detection accuracy

**Future Improvements** (Planned):
- [ ] Increase "Angry" class accuracy to 90%+
- [ ] Add more emotion classes (Fear, Playful, Neutral)
- [ ] Improve small dog detection
- [ ] Optimize for mobile devices
- [ ] Multi-dog tracking capability

---

## 🚀 Inference Optimization

### GPU Inference
```python
# Use GPU for faster inference
model = YOLO("models/emotion_detect.pt")
model.to('cuda')  # Move to GPU
results = model(image, device=0)  # Run on GPU 0
```

### Batch Processing
```python
# Process multiple images at once
images = [img1, img2, img3]
results = model(images, batch=3)
```

### Reduce Confidence Threshold
```python
# Lower threshold for more detections (trade-off: more false positives)
results = model(image, conf=0.25)  # Default is 0.5
```

---

## 📈 Performance Benchmarks

### Inference Speed (NVIDIA Tesla T4)

| Model | Resolution | FPS | Latency |
|-------|-----------|-----|---------|
| Dog Detection | 640x640 | 35 | 29ms |
| Emotion Detection | 640x640 | 40 | 25ms |
| **Combined Pipeline** | 640x640 | **30** | **33ms** |

### Inference Speed (CPU - Intel i7)

| Model | Resolution | FPS | Latency |
|-------|-----------|-----|---------|
| Dog Detection | 640x640 | 8 | 125ms |
| Emotion Detection | 640x640 | 10 | 100ms |
| **Combined Pipeline** | 640x640 | **7** | **143ms** |


---

## 🔐 License

These models are released under the MIT License for academic and non-commercial use.

For commercial use, please contact: [rezzak.eng@gmail.com](mailto:rezzak.eng@gmail.com)

---

## 🤝 Contributing

To improve the models:

1. Collect additional training data (especially "Angry" class)
2. Report issues or edge cases
3. Suggest architectural improvements
4. Submit pull requests with enhancements

---

## 📧 Support

For model-related questions:

**Abdurrezzak ŞIK**
- 📧 Email: [rezzak.eng@gmail.com](mailto:rezzak.eng@gmail.com)
- 🐙 GitHub: [@Ai-rezzak](https://github.com/Ai-rezzak)

---

## 🙏 Acknowledgments

- **TÜBİTAK 2209-A** for research funding
- **Ultralytics** for YOLOv8 framework
- **Google Colab** for GPU resources
- **PyTorch** team for deep learning framework

---

<p align="center">
  <i>"Two models working together to understand our dogs"</i> 🐕🤖
</p>