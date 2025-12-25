# 📊 Dataset Information

This directory contains sample data and documentation for the Dog Emotion Detection project.

---

## 📁 Directory Structure

```
data/
├── sample_images/          # Sample dog images for testing
├── annotations/            # Example annotation files (YOLO format)
└── README.md              # This file
```

---

## 🎯 Dataset Overview

### Emotion Detection Dataset

The emotion detection model was trained on **4,806 labeled images** across 4 emotion classes:

| Emotion Class | Training Samples | Description |
|--------------|------------------|-------------|
| **Happy** 😊 | 1,543 images | Joyful, playful, tail wagging |
| **Angry** 😠 | 912 images | Aggressive, teeth showing, defensive |
| **Sad** 😢 | 1,083 images | Depressed, low energy, withdrawn |
| **Sleeping** 😴 | 1,268 images | Resting, eyes closed, relaxed |
| **Total** | **4,806 images** | - |

### Dog Detection Dataset

The dog detection model was trained on **14,065 dog images** to accurately identify dogs and filter out other animals.

---

## 📥 Data Collection Sources

Data was collected from three primary sources:

### 1. Online Repositories
- **Google Images**: Public domain dog photos
- **Kaggle Open Images v7**: Large-scale annotated dataset
- **Open-source databases**: Various CC-licensed repositories

### 2. Veterinary Clinics
- Expert-provided images from real clinical cases
- High-quality emotion state documentation
- Professional assessment of emotional states

### 3. Custom Photography
- Self-captured images in controlled environments
- Natural behavior in various settings
- Permission obtained from dog owners

---

## 🏷️ Data Labeling Process

### Tools Used
- **MakeSense.ai**: Online annotation platform
- **YOLO Format**: Bounding box annotations

### Annotation Format

Each image has a corresponding `.txt` file with YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

**Example** (`image_001.txt`):
```
0 0.516 0.438 0.312 0.425
```

Where:
- `class_id`: 0=Happy, 1=Angry, 2=Sad, 3=Sleeping
- `x_center, y_center`: Normalized center coordinates (0-1)
- `width, height`: Normalized box dimensions (0-1)

---

## 📊 Data Split

The dataset was split into three sets:

| Split | Percentage | Purpose |
|-------|-----------|---------|
| **Training** | 80% | Model training |
| **Validation** | 10% | Hyperparameter tuning |
| **Test** | 10% | Final evaluation |

---

## 🔄 Data Augmentation

Data augmentation was performed using **Roboflow** with the following techniques:

### Augmentation Techniques Applied:
- ✅ **Rotation**: ±15 degrees
- ✅ **Horizontal Flip**: 50% probability
- ✅ **Brightness**: ±20% adjustment
- ✅ **Contrast**: ±15% adjustment
- ✅ **Blur**: Slight gaussian blur (0-1.5px)
- ✅ **Noise**: Up to 1% random noise

### Benefits:
- Improved model generalization
- Balanced class distribution
- Robustness to different lighting conditions
- Better performance on diverse dog breeds

---

## 🐕 Dog Breeds Represented

The dataset includes various dog breeds to ensure model generalization:

- **Small breeds**: Chihuahua, Pomeranian, Yorkshire Terrier
- **Medium breeds**: Beagle, Bulldog, Cocker Spaniel
- **Large breeds**: German Shepherd, Golden Retriever, Labrador
- **Giant breeds**: Great Dane, Saint Bernard
- **Mixed breeds**: Various crossbreeds

---

## 📋 Image Specifications

### Emotion Detection Dataset
- **Format**: JPG, PNG
- **Resolution**: Various (640x640 after preprocessing)
- **Color**: RGB
- **Quality**: High resolution, well-lit images
- **Total Size**: ~2.5 GB (uncompressed)

### Dog Detection Dataset
- **Format**: JPG, PNG
- **Resolution**: Various (640x640 after preprocessing)
- **Color**: RGB
- **Quality**: Diverse quality for robustness
- **Total Size**: ~8 GB (uncompressed)

---

## 🚫 Dataset Limitations

### Known Issues:
- **Class Imbalance**: "Angry" class has fewer samples (912 vs 1,543 for "Happy")
- **Breed Bias**: Some breeds more represented than others
- **Background Variety**: Most images taken in indoor/clinical settings
- **Lighting Conditions**: Limited low-light scenarios

### Recommendations:
- Collect more "Angry" emotion samples
- Include more diverse breeds
- Add outdoor and varied background images
- Include different time-of-day lighting

---

## 📥 Download Full Dataset

Due to GitHub's file size limitations, the complete dataset is not included in this repository.

### Option 1: Download from Cloud Storage

🔗 [Download Complete Dataset (Google Drive)](https://drive.google.com/your-link-here)

**Contents**:
- Full training dataset (80%)
- Validation dataset (10%)
- Test dataset (10%)
- Annotation files (YOLO format)

### Option 2: Request Access

For academic research or collaboration:
📧 Contact: [rezzak.eng@gmail.com](mailto:rezzak.eng@gmail.com)

---

## 🔐 Data Usage & Ethics

### Usage Rights
- ✅ Academic research
- ✅ Non-commercial projects
- ✅ Model training and testing
- ❌ Commercial use without permission
- ❌ Redistribution without attribution

### Ethical Considerations
- All images collected with owner consent
- No identifiable personal information included
- Veterinary images used with professional approval
- Animal welfare prioritized during data collection

---

## 📊 Sample Images

This repository includes a small subset of sample images in `sample_images/` for:
- Quick testing
- Documentation purposes
- Demo examples

**Sample structure**:
```
sample_images/
├── happy/
│   ├── happy_001.jpg
│   ├── happy_002.jpg
│   └── ...
├── angry/
│   ├── angry_001.jpg
│   └── ...
├── sad/
│   └── ...
└── sleeping/
    └── ...
```

---

## 🏷️ Annotation Examples

Example annotations are provided in `annotations/` directory:

```
annotations/
├── example_001.txt        # YOLO format annotation
├── example_002.txt
└── classes.txt            # Class names file
```

**classes.txt**:
```
Happy
Angry
Sad
Sleeping
```

---

## 📈 Dataset Statistics

### Emotion Detection

| Metric | Value |
|--------|-------|
| Total Images | 4,806 |
| Total Annotations | 4,806 |
| Average Dogs per Image | 1.0 |
| Image Resolution (avg) | 1920x1080 |
| Annotation Quality | 98.5% |

### Class Distribution

```
Happy:    32.1% (1,543 images) ████████████████
Angry:    19.0% (912 images)   █████████
Sad:      22.5% (1,083 images) ███████████
Sleeping: 26.4% (1,268 images) █████████████
```

---

## 🛠️ Data Preprocessing

Before training, images underwent preprocessing:

1. **Resize**: All images resized to 640x640
2. **Normalization**: Pixel values normalized to [0, 1]
3. **Format Conversion**: Converted to RGB if needed
4. **Quality Check**: Blurry or low-quality images removed
5. **Annotation Validation**: Verified bounding box coordinates

---

## 📝 Citation

If you use this dataset, please cite:

```bibtex
@dataset{sik2023dog_dataset,
  title={Dog Emotion Detection Dataset},
  author={ŞIK, Abdurrezzak},
  institution={TÜBİTAK 2209-A, Dicle University},
  year={2023},
  note={4,806 labeled images across 4 emotion classes}
}
```

---

## 🤝 Contributing

To contribute to the dataset:

1. Collect high-quality dog images
2. Ensure clear emotional state visibility
3. Label using YOLO format
4. Submit via pull request or email

**Guidelines**:
- Clear, well-lit images
- Single dog per image (preferred)
- Visible facial features
- Diverse breeds and ages
- Ethical collection methods

---

## 📧 Contact

For questions about the dataset:

**Abdurrezzak ŞIK**
- 📧 Email: [rezzak.eng@gmail.com](mailto:rezzak.eng@gmail.com)
- 🐙 GitHub: [@Ai-rezzak](https://github.com/Ai-rezzak)

---

## 🙏 Acknowledgments

- Veterinary clinics for providing professional images
- Dog owners for allowing photography
- TÜBİTAK 2209-A for funding support
- MakeSense.ai for annotation platform
- Roboflow for augmentation tools

---

<p align="center">
  <i>"Every image tells a story about our furry friends"</i> 🐕❤️
</p>