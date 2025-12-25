# Model Information

This directory contains the trained deep learning models developed within the
**TÜBİTAK 2209-A – Dog Body Language and Emotion Analysis Project**.

The models were trained using the **YOLOv8 object detection framework**
and are designed for real-time inference on images and videos.

---

## Model Overview

Two independent models were developed and evaluated:

### 1️⃣ Dog Detection Model
- **Purpose:** Detect the presence and location of dogs in images and video streams
- **Model Type:** YOLOv8 Object Detection
- **Input:** RGB images / video frames
- **Output:** Bounding boxes for detected dogs
- **Performance:**  
  - Mean Accuracy: **~89.4%**

---

### 2️⃣ Emotion Detection Model
- **Purpose:** Classify and localize dog emotions based on body posture and facial cues
- **Model Type:** YOLOv8 Object Detection
- **Target Classes:**
  - Happy
  - Angry
  - Sad
  - Sleeping
- **Input:** Cropped dog regions or full images
- **Output:** Bounding boxes with emotion labels
- **Performance:**  
  - Mean Accuracy: **~92.3%**

---

## Training Details

- **Framework:** Ultralytics YOLOv8
- **Programming Language:** Python
- **Training Environment:** GPU-accelerated environment
- **Image Resolution:** 640 × 640
- **Optimization:** Default YOLOv8 optimization strategies
- **Loss Functions:** Standard YOLOv8 detection losses

Hyperparameters were tuned experimentally to balance accuracy and inference speed.

---

## Model Files

The trained weight files (`.pt`) are **not included in this public repository**
due to file size limitations and distribution policies.

To obtain the trained models:
- Please contact the project author directly
- Or refer to the download links provided upon request

---

## Ethical and Responsible Use

These models are intended **for educational and research purposes only**.

Important notes:
- The models are **not designed for medical, safety-critical, or commercial use**
- Predictions may contain errors and should not be used as the sole basis
  for decision-making
- The dataset was prepared with expert assistance to minimize annotation bias,
  but limitations still apply

---

## Citation and Acknowledgment

If you use or reference this project in academic or research work,
please acknowledge the support of **TÜBİTAK 2209-A** and the project contributors.

---

## Contact

For model access, questions, or collaboration opportunities:

**Project Author:** Abdurrezzak Şık  
**Institution:** Dicle University  
**Project Type:** TÜBİTAK 2209-A Undergraduate Research Project
