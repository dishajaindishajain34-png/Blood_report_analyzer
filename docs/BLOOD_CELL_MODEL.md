# Blood Cell Detection Model — Documentation

## Overview

HemaVision AI uses a **YOLOv8n** (nano) object detection model to classify and locate blood cells in peripheral blood smear microscope images. The model detects three classes simultaneously:

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | Platelet | Thrombocytes small anucleate cell fragments |
| 1 | RBC | Red Blood Cells (Erythrocytes) |
| 2 | WBC | White Blood Cells (Leukocytes) |

---

## Dataset

### Source
- **Name**: BCCD Dataset (Blood Cell Count and Detection)
- **Author**: Shenggan
- **Repository**: https://github.com/Shenggan/BCCD_Dataset
- **License**: MIT

### Raw Dataset Statistics
| Class | Total Instances |
|-------|----------------|
| RBC | 4,155 |
| WBC | 372 |
| Platelet | 361 |
| **Total images** | **364** |

### Train/Val/Test Split (70/15/15)
| Split | Images (before aug.) | Images (after aug.) |
|-------|---------------------|---------------------|
| Train | 254 | 674 |
| Val | 54 | 54 |
| Test | 56 | 56 |

### Annotation Counts After Split
| Split | Platelet | RBC | WBC |
|-------|---------|-----|-----|
| Train | 1,008 | 7,801 | 704 |
| Val | 51 | 619 | 53 |
| Test | 58 | 632 | 56 |

### Augmentation Strategy
Platelet-containing training images received 3x geometric augmentations (hflip, vflip, both) to address class imbalance.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | YOLOv8n (fine-tuned from existing best.pt) |
| Epochs | 3 |
| Image size | 320x320 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 0.005 |
| Hardware | CPU (AMD Athlon Silver 3050U) |
| Training time | ~19 minutes (1152.6s) |

---

## Model Performance (Held-out Test Split: 56 images, 746 instances)

### New Model (Deployed)
| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| Platelet | 72.47% | 81.03% | 82.74% | 40.38% |
| RBC | 81.35% | 75.90% | 85.61% | 55.73% |
| WBC | 96.81% | 100.00% | 99.06% | 73.76% |
| Overall | 83.54% | 85.65% | 89.14% | 56.62% |

### Old vs New Comparison
| Class | Old mAP50 | New mAP50 | Change |
|-------|-----------|-----------|--------|
| Platelet | 0.00% | 82.74% | +82.74pp |
| RBC | 89.10% | 85.61% | -3.49pp |
| WBC | 99.48% | 99.06% | -0.42pp |
| Overall | 62.86% | 89.14% | +26.28pp |

---

## Model File
- Production model: models/yolov8_blood/best.pt
- Backup: models/yolov8_blood/best_platelet_improved.pt
- Architecture: YOLOv8n - 73 layers, 3,006,233 parameters, 8.1 GFLOPs
- Class mapping: {0: Platelet, 1: RBC, 2: WBC}

---

## Known Limitations
1. Small Platelet detection: platelets are tiny; higher resolution training (640+) with GPU would improve recall further.
2. RBC slight regression: ~3.5pp mAP50 drop from fine-tuning on BCCD-only with 3 epochs.
3. CPU inference speed: ~317ms per image on AMD Athlon Silver 3050U.
4. WBC subtype classifier: no trained weights; falls back to WHO-ratio estimation.
5. Anemia model data leakage: label is Hb-threshold-derived; disclosed in UI.

---

*Last updated: 2026-08-29 | Model: YOLOv8n fine-tuned on BCCD (3 epochs, imgsz=320)*
