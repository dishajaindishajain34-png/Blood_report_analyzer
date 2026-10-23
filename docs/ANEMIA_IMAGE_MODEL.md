# Image-Based Anemia Screening Model (MobileNetV3-Small)

*HemaVision AI — Deep Learning Photomicrography Classifier*

---

## 1. Overview & Clinical Context

The Image-Based Anemia Model is a deep Convolutional Neural Network (MobileNetV3-Small) fine-tuned on high-resolution peripheral blood smear microscopy images (combining the AneRBC and Chula-RBC-12 datasets). It analyzes erythrocyte (RBC) morphological abnormalities—such as microcytes, macrocytes, spherocytes, target cells, and central pallor hypochromia—to evaluate the visual likelihood of anemia directly from microscopy slides.

---

## 2. Model Audit: Root-Cause Analysis of 5.0% Specificity Failure Mode

In earlier runs, the model exhibited an extreme diagnostic failure mode:
- **Sensitivity (Recall)**: 97.73% (86 TP, 2 FN)
- **Specificity**: **5.00%** (1 TN, 19 FP)
- **Root Causes**:
  1. **Extreme Class Imbalance**: Training set had ~5:1 ratio of Anemic to Normal slides (585 training images with 499 Anemic vs 86 Normal).
  2. **Unweighted Loss**: Training with standard unweighted `CrossEntropyLoss` strongly incentivized predicting the majority class (Anemic).
  3. **Hardcoded 0.5 Threshold**: The uncalibrated softmax probability output for Normal slides averaged 0.60–0.75, so a rigid 0.5 threshold misclassified 95% of healthy slides as anemic.
  4. **Unstratified Splitting**: Random shuffling resulted in uneven distribution of Normal cases across validation and test splits.

---

## 3. Engineering Interventions Applied

1. **Stratified Patient-Level Group Split**:
   - Total deduplicated images: **837** (124 Normal, 713 Anemic).
   - Stratified at patient/sample group level:
     - **Train Split (70%)**: 585 images (86 Normal, 499 Anemic | 585 patients)
     - **Validation Split (15%)**: 124 images (18 Normal, 106 Anemic | 124 patients)
     - **Held-out Test Split (15%)**: 128 images (20 Normal, 108 Anemic | 128 patients)
2. **Inverse Class Frequency Weighting**:
   $$w_0 = \frac{N_{\text{train}}}{2 \cdot N_0} = 3.4012, \quad w_1 = \frac{N_{\text{train}}}{2 \cdot N_1} = 0.5862$$
   Enforced in `nn.CrossEntropyLoss(weight=class_weights)`.
3. **Threshold Calibration via Youden's J Statistic**:
   Operating threshold calibrated on the **validation split only** using Youden's Index ($J = \text{Sensitivity} + \text{Specificity} - 1$):
   $$\tau^* = 0.8500 \quad (\text{Validation } J = 0.6352)$$

---

## 4. Performance Metrics (Held-Out Test Split — 128 Images)

The table below contrasts the default decision threshold against the calibrated operating threshold on the untouched test split:

| Metric | Unweighted (Baseline) | Default Threshold ($\tau = 0.50$) | Calibrated Operating ($\tau^* = 0.85$) |
|---|:---:|:---:|:---:|
| **Test Accuracy** | 80.56% | 83.59% | **75.78%** |
| **Sensitivity (Recall)** | 97.73% | 92.59% | **75.93%** |
| **Specificity** | **5.00%** | 35.00% | **75.00%** (+70% absolute) |
| **Precision** | 81.90% | 88.50% | **94.25%** |
| **F1-Score** | 0.8912 | 0.9050 | **0.8410** |
| **ROC AUC** | 0.7477 | 0.8500 | **0.8500** (+0.1023) |

### Confusion Matrix on Held-Out Test Set (Calibrated Threshold $\tau^* = 0.85$)

```
                        Predicted Normal       Predicted Anemic
Actual Normal (20)             15 (TN)                 5 (FP)
Actual Anemic (108)            26 (FN)                82 (TP)

Specificity: 15 / 20 = 75.00%
Sensitivity: 82 / 108 = 75.93%
```

---

## 5. Model Artifacts

- **Model Weights**: `models/anemia_image/mobilenetv3_anemia.pth`
- **Metadata & History**: `models/anemia_image/model_metadata.json`
- **Quantitative Metrics**: `models/anemia_image/metrics.json`
- **Class Mapping**: `models/anemia_image/class_mapping.json`
- **Preprocessing Configuration**: `models/anemia_image/preprocessing_config.json`
- **Visualizations**:
  - `models/anemia_image/training_history.png`
  - `models/anemia_image/confusion_matrix.png`
  - `models/anemia_image/roc_curve.png`
