# Technical Project Report
## AI-Based Smart Blood Report Analyzer (HemaVision AI)

---

### VERIFIED FACTS vs INFERENCES vs MISSING INFORMATION

| Claim | Classification | Evidence |
|---|---|---|
| **YOLOv8 blood cell detector class mapping is 0: Platelet, 1: RBC, 2: WBC** | **VERIFIED** | Explicitly defined in `data/blood_cells/data.yaml` and loaded in `src/detect.py` (`CLASS_NAMES`). |
| **YOLOv8 final test performance: mAP50 = 89.14%, mAP50-95 = 56.62%** | **VERIFIED** | Stored in `models/yolov8_blood/metrics.json` under `"new_model"`. |
| **YOLOv8 training run blood_cell_detector-7 early stopped at epoch 16** | **VERIFIED** | Documented in `TRAINING_REPORT.md` (Epochs completed: 16, mAP50: 74.31%). |
| **Image-based anemia model (MobileNetV3-Small) trained on AneRBC dataset** | **VERIFIED** | Confirmed by `scripts/train_anemia_image.py` and `models/anemia_image/model_metadata.json`. |
| **Image anemia model has 80.56% accuracy but only 5.0% specificity** | **VERIFIED** | Explicitly recorded in `models/anemia_image/metrics.json`. |
| **CBC anemia Random Forest model achieved 100% accuracy on validation and test sets** | **VERIFIED** | Documented in `models/anemia/metrics.json`. |
| **CBC model target label has 95.0% concordance with WHO Hemoglobin thresholds** | **VERIFIED** | Checked by `scripts/train_anemia_model.py` and log in `DATA_LEAKAGE_CHECK.md`. |
| **Hemoglobin is the most important feature (83.32%) in the CBC Random Forest model** | **VERIFIED** | Recorded in `models/anemia/metrics.json` under `"feature_importances"`. |
| **Trained fusion model weights (fusion_model.joblib) are missing** | **VERIFIED** | Loading skips and falls back to probability averaging (warning logged in `src/anemia_fusion.py`). |
| **WBC classifier (wbc_classifier_sklearn.pth) exists but falls back to WHO ratios on empty crops** | **VERIFIED** | Explicit fallback logic implemented in `src/classify_wbc.py`. |
| **RGB/BGR channel swap in YOLO NumPy arrays was causing misclassifications** | **VERIFIED** | Documented code comments and conversion logic in `src/detect.py` (`cv2.cvtColor` call). |
| **Analysis records are persisted in SQLite DB at data/blood_analyzer.db** | **VERIFIED** | Configured in `database/connection.py` and models in `database/models.py`. |
| **PDF reports are generated asynchronously using fpdf2** | **VERIFIED** | Scheduled as a background task in `backend/routes/analyze.py` calling `src/report_generator.py`. |
| **Streamlit frontend (app.py) was built for Phase 1** | **VERIFIED** | File `frontend/app.py` exists in root and documented in `README.md`. |
| **Vite/React frontend is built for Phase 2** | **VERIFIED** | Project exists under `ui/` directory with `package.json` and React code. |

---

# CHAPTER 1 — Project Audit & Introduction
This project, titled **HemaVision AI (AI-Based Smart Blood Report Analyzer)**, is an automated clinical decision support system. It processes blood smear microscope images to count cells (Red Blood Cells, White Blood Cells, and Platelets), estimate WBC differentials, and screen for anemia using a hybrid multimodal fusion pipeline combining computer vision and machine learning.

### 1.1 Project Objective
The primary objective of the system is to automate cell enumeration and provide preliminary anemia screening from a single microscopy field of view. By combining visual features from blood smear images and numerical inputs from laboratory CBC counts, the system offers an assistive diagnostic stream that flags abnormalities relative to World Health Organization (WHO) reference ranges.

### 1.2 Motivation
Traditional manual microscopy is labor-intensive and subjective, requiring specialized hematologists to count cells grid-by-grid. While automated hematology analyzers (flow cytometers) are the clinical gold standard, they do not provide morphological visualizations and are expensive. HemaVision AI offers a low-cost, digital alternative that performs rapid count validation, leukocyte differential profiling, and multimodal anemia screening directly from digital microscope images.

### 1.3 Tech Stack & Architecture
- **Programming Languages**: Python 3.11+, TypeScript (ES6+)
- **Frontend Stack**: React 19, Vite 8, TanStack Router (v1.170.18), TanStack Query (v5.101.1), Tailwind CSS 4, Framer Motion, Recharts
- **Backend Stack**: FastAPI (v0.110.0), Uvicorn (v0.29.0), SQLAlchemy ORM
- **Database**: SQLite (local file at `data/blood_analyzer.db`)
- **Computer Vision & ML**: OpenCV (v4.9.0), Ultralytics YOLOv8 (Nano), PyTorch (v2.1.0), Scikit-Learn (v1.9.0), joblib
- **PDF Generation**: fpdf2 (v2.7.9) and Matplotlib (v3.8.0)

---

# CHAPTER 2 — System Pipeline & Architecture
The system employs a dual-stream architecture that processes visual and laboratory data in parallel before fusing them into a final diagnostic verdict.

```
                  [USER INPUT]
         (Smear Image & Optional CBC)
                      │
                      ▼
               [REACT FRONTEND]
                      │
                      ▼
               (POST /analyze)
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
  [IMAGE PIPELINE]          [CBC PIPELINE]
  (FastAPI Backend)         (FastAPI Backend)
         │                         │
         ├─► Gaussian Blur         └─► Extract params
         ├─► CLAHE (Contrast)      └─► Validate range
         ├─► YOLOv8 Detection      └─► Scale features
         │   (RBC/WBC/Platelet)    └─► Random Forest
         │   (Optional Tiling)     └─► CBC Probability
         │                         │
         ├─► WBC Crops             │
         │   (ResNet50 Classifier) │
         │   (Fallback WHO Ratios) │
         │                         │
         └─► MobileNetV3-Small     │
             Image Probability     │
         │                         │
         ▼                         ▼
     [ANEMIA IMAGE PROB]        [ANEMIA CBC PROB]
         │                         │
         └────────────┬────────────┘
                      ▼
               [MULTIMODAL FUSION]
          (Trained Logistic Regression)
            (Average Fallback: 50/50)
                      │
                      ▼
               [FINAL VERDICT]
          (Normal / Anemic / Grade)
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
  [SQLITE RECORD]           [PDF GENERATOR]
  (data/blood_analyzer.db)   (Asynchronous Task)
```

### 2.1 Component Specifications
1. **Preprocessing (`src/preprocess.py:preprocess`)**: Applies Gaussian blur (3x3 kernel) to reduce high-frequency noise, converts to LAB color space, applies CLAHE (ClipLimit=2.0, Grid=8x8) to the L-channel for contrast normalization, and resizes to 640x640.
2. **Cell Detection (`src/detect.py:detect_cells`)**: Runs YOLOv8 Nano (`models/yolov8_blood/best.pt`). If zero RBCs are detected (indicating a dense smear), it triggers `_detect_cells_tiled` using a 320x320 sliding window with 50% overlap and cross-tile Non-Maximum Suppression (NMS, IOU=0.50, Conf=0.25).
3. **WBC Classification (`src/classify_wbc.py:classify_wbc_crops`)**: Extracts padding crops around WBCs and runs a 224-dim feature extractor (RGB + HSV + Gradient histograms). Classifies crops using an ensemble classifier (`wbc_classifier_sklearn.pth`). Falls back to WHO healthy blood ratios if model weights are missing.
4. **Anemia Models**:
   - **Image-Based (`src/anemia_image_model.py`)**: Runs MobileNetV3-Small on the preprocessed image.
   - **CBC-Based (`src/anemia_model.py`)**: Runs a Random Forest model on 5 standard parameters (Gender, Hemoglobin, MCV, MCH, MCHC).
5. **Multimodal Fusion (`src/anemia_fusion.py`)**: Combines probabilities. Falls back to simple averaging if the trained Logistic Regression model is unavailable.
6. **Report Generation (`src/report_generator.py`)**: Asynchronously generates a clinical-grade PDF with metadata, image thumbnails, cell counts, WBC differential pie chart, and diagnostic indicators.
7. **Database Storage (`database/crud.py`)**: Inserts results into the `analysis_records` table in SQLite.

---

# CHAPTER 3 — Datasets & Splitting

### 3.1 BCCD Dataset (Blood Cell Detection)
Used to train the YOLOv8 object detector. In this repository, the dataset was augmented and structured into:
- **Total Images**: 784
- **Splits**: Train (674 images), Validation (54 images), Test (56 images)
- **Annotations**: RBC (9,052 boxes), WBC (813 boxes), Platelet (1,117 boxes)

### 3.2 AneRBC Dataset (Image-Based Anemia)
Used to train the MobileNetV3-Small image classifier.
- **Total Images**: 738. After duplicate removal, 716 unique images.
- **Class Balance**: Anemic (613 images), Normal (125 images) — indicating severe class imbalance.
- **Splits**: Train (501 images), Validation (107 images), Test (108 images)

### 3.3 Kaggle Anemia Dataset (CBC-Based Anemia)
Used to train the Random Forest tabular classifier.
- **Total Records**: 1,421
- **Class Balance**: Normal (801 records, 56.4%), Anemic (620 records, 43.6%)
- **Splits**: Train (994 records, 70%), Validation (213 records, 15%), Test (214 records, 15%)
- **Features**: `Gender`, `Hemoglobin`, `MCH`, `MCHC`, `MCV`

---

# CHAPTER 4 — Model Training & Performance

### 4.1 YOLOv8 Blood Cell Detector
Trained on the augmented BCCD dataset.
- **Architecture**: YOLOv8 Nano (Ultralytics)
- **Epochs**: 20 configured, early stopped at 16.
- **Performance (Held-out Test Split of 56 images)**:
  - **Overall**: Precision = 83.54%, Recall = 85.65%, mAP50 = 89.14%, mAP50-95 = 56.62%
  - **Platelets**: Precision = 72.47%, Recall = 81.03%, mAP50 = 82.74%
  - **RBC**: Precision = 81.35%, Recall = 75.90%, mAP50 = 85.61%
  - **WBC**: Precision = 96.81%, Recall = 100.0%, mAP50 = 99.06%
- **Weight File**: `models/yolov8_blood/best.pt` (5.94 MB)

### 4.2 MobileNetV3-Small (Image Anemia Model)
Trained on the AneRBC dataset.
- **Architecture**: MobileNetV3-Small (PyTorch backbone)
- **Epochs**: 5 completed.
- **Performance (Test Set)**:
  - **Accuracy**: 80.56%
  - **Precision**: 81.90%
  - **Recall (Sensitivity)**: 97.73%
  - **Specificity**: 5.0%
  - **F1-Score**: 89.12%
  - **ROC-AUC**: 74.77%
  - **Confusion Matrix**: True Negative = 1, False Positive = 19, False Negative = 2, True Positive = 86
- **Weight File**: `models/anemia_image/mobilenetv3_anemia.pth` (6.2 MB)
- **Clinical Limitation**: The model exhibits a severe bias towards positive predictions (97.73% recall but only 5.0% specificity). This is caused by the extreme class imbalance in the AneRBC training dataset (613 Anemic vs. 125 Normal).

### 4.3 Random Forest (CBC Anemia Model)
Trained on the Kaggle Anemia dataset.
- **Architecture**: Random Forest Classifier (`n_estimators=100`, `class_weight='balanced'`)
- **Performance (Held-out Test Set)**:
  - **Accuracy / Precision / Recall / F1 / Specificity**: 1.0 (100.0%)
  - **ROC-AUC**: 1.0
  - **Confusion Matrix**: TN = 121, FP = 0, FN = 0, TP = 93
- **Feature Importances**: `Hemoglobin` = 83.32%, `Gender` = 9.82%, `MCV` = 2.77%, `MCH` = 2.23%, `MCHC` = 1.86%
- **Critical Methodological Limitation (Target Leakage)**: The perfect classification score (100.0%) is due to **target leakage**. The target label (`Result`) in the Kaggle dataset was derived directly from standard Hemoglobin and Gender thresholds (95% concordance rule). Since Hemoglobin is also provided as an input feature, the classifier trivially learns this rule (evidenced by 83.32% Hemoglobin importance). This model is not an independent diagnostic tool, but a threshold-reproduction mechanism.

### 4.4 Multimodal Fusion Model
- **Trained Model**: Logistic Regression model. The weight file `models/anemia_fusion/fusion_model.joblib` was not available in the inspected repository due to the absence of a paired image-CBC training dataset.
- **Current Strategy**: Probability Averaging fallback.
  $$	ext{Fused Probability} = 0.5 	imes P(	ext{Anemic})_{	ext{image}} + 0.5 	imes P(	ext{Anemic})_{	ext{CBC}}$$

---

# CHAPTER 5 — Development History & Major Bug Fixes

Throughout the project, several critical issues were discovered and resolved:

### 5.1 RGB/BGR Channel-Order Bug
- **Problem**: YOLOv8 returned highly incorrect counts (RBCs misclassified as WBCs).
- **Cause**: OpenCV loads images as BGR, but PIL loads them as RGB. Passing a raw RGB NumPy array to YOLO caused red and blue channels to swap.
- **Solution**: Pass the PIL Image directly to YOLO (it handles RGB natively) or convert the NumPy array from RGB to BGR (`cv2.cvtColor(image, cv2.COLOR_RGB2BGR)`) prior to inference.
- **File Changed**: `src/detect.py`

### 5.2 Multipart/Form-Data Parsing Bug
- **Problem**: CBC inputs were always received as `None` on the backend, skipping the ML stream.
- **Cause**: FastAPI parses multipart bodies once. Calling `await request.form()` a second time in the route handler returned empty dictionary because the stream was already exhausted.
- **Solution**: Declared all five CBC parameters as explicit `Form(None)` variables in the endpoint signature.
- **File Changed**: `backend/routes/analyze.py`

### 5.3 Anemia Panel JSX Nesting Error
- **Problem**: UI results rendering was broken with React syntax errors.
- **Cause**: An incorrect code replacement nested the `AnemiaMLPanel` component inside the JSX of `ImageAnemiaPanel` and left orphaned tags.
- **Solution**: Restructured `AnemiaMLPanel` to decouple CBC inputs rendering from the ML prediction, cleaned up JSX tags, and verified with typescript compilation.
- **File Changed**: `ui/src/routes/_app.results.$analysisId.tsx`

---

# CHAPTER 6 — Reproducibility Instructions
To run this project on a local Windows laptop:

1. **System Requirements**: Python 3.11+, Node.js 18+
2. **Environment Setup**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Database Initialization**:
   The SQLite database table is initialized automatically on backend startup.
4. **Model Weights Verification**:
   Ensure the following weights exist in the `models/` directory:
   - `models/yolov8_blood/best.pt`
   - `models/anemia_image/mobilenetv3_anemia.pth`
   - `models/anemia/anemia_model.joblib`
   - `models/resnet50_wbc/wbc_classifier_sklearn.pth`
5. **Start Backend**:
   ```bash
   $env:PYTHONPATH="."
   python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
   ```
6. **Start Frontend**:
   ```bash
   cd ui
   npm install
   npm run dev -- --host 0.0.0.0 --port 8083
   ```
7. **Run Unit Tests**:
   ```bash
   .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
   ```

---

# CHAPTER 7 — Medical & Ethical Disclaimer
This system is intended as an **assistive diagnostic screening tool only**. The blood cell counts, WBC differentials, and anemia screening results are preliminary estimates. They are **not** equivalent to laboratory flow cytometry counts or clinical hematologist reviews. The system must never be used to override certified pathologist reviews or make therapeutic decisions.
