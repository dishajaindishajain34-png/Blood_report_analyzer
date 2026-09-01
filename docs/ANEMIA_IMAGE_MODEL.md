# Image-Based Anemia Screening Model (MobileNetV3-Small)

*HemaVision AI — Deep Learning Photomicrography Classifier*

---

## 1. Overview

The Image-Based Anemia Model is a deep Convolutional Neural Network (MobileNetV3-Small) fine-tuned on high-resolution peripheral blood smear microscopy images from the Chula-RBC-12 dataset. It analyzes RBC morphology (microcytes, macrocytes, spherocytes, target cells, hypochromia) to directly predict the likelihood of anemia from a single blood-smear image.

---

## 2. Model Architecture & Training

- **Backbone**: MobileNetV3-Small (pretrained on ImageNet)
- **Input Resolution**: 224 x 224 x 3 RGB
- **Output**: Binary softmax probability `[P(Normal), P(Anemia)]`
- **Class Mapping**:
  - `0` = Normal / Healthy
  - `1` = Anemic
- **Optimizer**: AdamW (lr=0.001, weight_decay=1e-4)
- **Loss Function**: CrossEntropyLoss
- **Split Strategy**: 70% Train / 15% Val / 15% Test (held-out)

---

## 3. Performance Metrics (Held-Out Test Split)

| Metric | Value |
|---|---|
| **Test Accuracy** | **80.56%** |
| **Sensitivity (Recall)** | **97.73%** |
| **F1-Score** | **0.8912** |
| **ROC AUC** | **0.7477** |

---

## 4. Confusion Matrix

```
True Positives (TP) : 86  (Anemic correctly identified)
False Negatives (FN): 2   (Anemic mischecked as Normal)
True Negatives (TN) : 1   (Normal correctly identified)
False Positives (FP): 19  (Normal flagged as Anemic)
```

---

## 5. Model Artifacts

- Model weights: `models/anemia_image/mobilenetv3_anemia.pth`
- Test metrics: `models/anemia_image/metrics.json`
- Class mapping: `models/anemia_image/class_mapping.json`
- Preprocessing config: `models/anemia_image/preprocessing_config.json`
