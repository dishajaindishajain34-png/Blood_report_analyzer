# MODEL_ARCHITECTURE.md — System Architecture & Dual-Stream Pipeline

This document illustrates the complete architectural design of the **AI-Based Smart Blood Report Analyzer**, highlighting the non-breaking Primary Blood Smear Image Pipeline and the new Secondary ML Anemia Detection Stream.

---

## 1. System Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                            PRIMARY PIPELINE (Blood Smear)                         |
|                                                                                   |
|  [ Blood Smear Image ] -> [ OpenCV CLAHE & Blur ] -> [ YOLOv8 (best.pt) ]         |
|                                                              |                    |
|                                                              v                    |
|                                              [ RBC / WBC / Platelet Detections ]  |
|                                                              |                    |
|                                                              v                    |
|                                              [ WBC Subtype Classifier (sklearn) ] |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
                                [ Image Cell Counts & Differential ]
                                       |
                                       +-----------------------+
                                                               |
+------------------------------------------------------+       |
|            SECONDARY PIPELINE (ML Anemia)            |       |
|                                                      |       |
|  [ User-Entered Lab CBC Inputs (Hb, MCV, MCH, Gender)]       |
|                         |                            |       |
|                         v                            |       |
|            [ StandardScaler Preprocessing ]           |       |
|                         |                            |       |
|                         v                            |       |
|           [ Trained Anemia ML Classifier ]            |       |
|                         |                            |       |
|                         v                            |       |
|        [ ML Anemia Prediction & Probability ]        |       |
+------------------------------------------------------+       |
                          |                                    |
                          v                                    v
       +---------------------------------------------------------------+
       |                     APPLICATION INTEGRATION                   |
       |                                                               |
       |   - Streamlit UI (Analyze, History, About Tabs)               |
       |   - FastAPI Backend (/analyze & REST endpoints)               |
       |   - FPDF2 Clinical PDF Generator (ANEMIA ML ANALYSIS section) |
       |   - SQLite Database Storage (data/blood_analyzer.db)          |
       +---------------------------------------------------------------+
```

---

## 2. Component Breakdown

### A. Primary Stream: Blood Smear Image Analysis
1. **Preprocessing (`src/preprocess.py`)**: Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) and Gaussian smoothing to enhance microscope images.
2. **Cell Detection (`src/detect.py`)**: Runs YOLOv8 (`models/yolov8_blood/best.pt`) to locate and bound Red Blood Cells (RBC), White Blood Cells (WBC), and Platelets.
3. **WBC Differential (`src/classify_wbc.py`)**: Crops WBC bounding boxes and classifies 5 subtypes (Neutrophil, Lymphocyte, Monocyte, Eosinophil, Basophil) using an sklearn ensemble classifier.
4. **Reference Range Assessment (`src/normal_ranges.py`)**: Compares image-derived cell density estimates against clinical reference ranges.

### B. Secondary Stream: ML Anemia Detection
1. **Model File (`models/anemia/anemia_model.joblib`)**: Random Forest classifier trained on 1,421 clinical CBC records.
2. **Inference Module (`src/anemia_model.py`)**: Accepts laboratory parameters (`Gender`, `Hemoglobin`, `MCV`, `MCH`, `MCHC`), validates inputs, scales features, and returns prediction (`"Normal"` vs `"Anemic"`), probability, and feature importances.
3. **Explicit User Input Policy (V1)**: The model processes user-entered laboratory CBC values. Image-derived cell counts are kept separate in Version 1.

### C. Unified Output & Storage
- **Streamlit UI (`frontend/app.py`)**: Displays image detections, reference-range findings, and secondary ML Anemia predictions in separate, clear panels.
- **PDF Report (`src/report_generator.py`)**: Renders image thumbnails, cell count tables, WBC differential pie chart, and an **ANEMIA ML ANALYSIS** block.
- **SQLite Database (`database/models.py`)**: Stores analysis records with backward-compatible schema fields (`anemia_ml_prediction`, `anemia_ml_probability`, `anemia_ml_model_version`, `cbc_inputs_json`).

---

## 3. Future Roadmap: Multimodal Fusion

If a future clinical dataset is acquired that contains paired blood smear microscope images alongside verified laboratory CBC blood panel measurements for the same patients, a joint multimodal feature fusion layer (concatenating image embeddings from YOLOv8/ResNet with tabular CBC embeddings) can be implemented.
