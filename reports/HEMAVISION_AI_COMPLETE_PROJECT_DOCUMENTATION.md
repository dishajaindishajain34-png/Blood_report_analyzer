# HemaVision AI — Complete Technical Project Documentation & System Knowledge Base
## Multimodal Artificial Intelligence for Preliminary Anemia Screening Using Blood Smear Photomicrographs and Complete Blood Count Parameters

---

### DOCUMENT CONTROL & SYSTEM METADATA
- **Project Title:** HemaVision AI — Multimodal AI-Based Anemia Screening Using Blood Smear Images and Complete Blood Count Parameters
- **System Version:** Production v2.0.0 (with Selectable Experimental GMU v1.0.0-exp)
- **Local Codebase Path:** `C:\Users\HP\Desktop\Blood_report_analyzer-main`
- **Remote Git Repository:** `https://github.com/dishajaindishajain34-png/Blood_report_analyzer`
- **Initial Upstream Commit:** `cbe5b3a` (*"Initial commit - Blood Report Analyzer"*, Tue Sep 1 09:20:56 2026 +0530)
- **Primary Runtime Frameworks:** Python 3.11+, PyTorch 2.1.0, Torchvision 0.16.0, Ultralytics YOLOv8 (v8.1.0), Scikit-Learn (v1.9.0), OpenCV (v4.9.0), FastAPI (v0.110.0), Streamlit (v1.32.0), React 19 (TypeScript / Vite 8 / TanStack Router)
- **Document Nature:** Comprehensive technical knowledge base, system architecture specification, dataset audit, and historical development record.

---

> [!CAUTION]
> ### MEDICAL SAFETY & ACADEMIC RESEARCH PROTOTYPE DISCLOSURE
> **HemaVision AI is an academic research prototype designed exclusively for preliminary screening exploration and educational research.**
> 1. **No Clinical Validation:** The system has not undergone prospective multi-center clinical trials, inter-laboratory reproducibility studies, or formal diagnostic validation on human patients. It **must not** be used as a standalone diagnostic tool, to direct clinical therapy, or to substitute for standard pathology workflows.
> 2. **Uncertified Software (SaMD Boundary):** The system has **not** been evaluated, cleared, or approved by the United States Food and Drug Administration (FDA) under 510(k), De Novo, or PMA pathways, nor does it possess CE mark certification under European Union Medical Device Regulation (EU MDR 2017/745). It has not undergone an accredited HIPAA security audit.
> 3. **Non-Linkage of Public Datasets:** Microscopic blood smear photomicrographs (`data/anerbc_raw/`, `chula_main.zip`) and tabular Complete Blood Count (CBC) records (`master_cbc_dataset.csv`) originate from separate, unlinked public repositories with **zero patient-level linkage**. 
> 4. **Synthetic Nature of Multimodal Fusion Results:** All multimodal fusion evaluations—including the Production Stacking Ensemble (99.44% test accuracy) and the Experimental Gated Multimodal Unit (97.50% test accuracy)—were evaluated on **synthetically paired combinations** of unlinked images and CBC records. These metrics demonstrate algorithmic convergence and feature-gating dynamics under artificial pairing rules; **they do not establish clinical diagnostic efficacy on real patients**.
> 5. **Mandatory Clinician Verification:** All computer vision cell detections, leukocyte differentials, and machine learning probabilities require independent review and confirmation by a board-certified hematologist or licensed clinical pathologist.

---

## TABLE OF CONTENTS

1. [Project Overview, Motivation & Clinical Boundaries](#1-project-overview-motivation--clinical-boundaries)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Complete Repository & Code Walkthrough](#3-complete-repository--code-walkthrough)
4. [Dataset History & Data Integrity Audit](#4-dataset-history--data-integrity-audit)
5. [Original Implementation & Baseline Results](#5-original-implementation--baseline-results)
6. [Chronological Development History & Major Modifications](#6-chronological-development-history--major-modifications)
7. [Image-Based Cytology & Classification Pipeline (YOLOv8 & MobileNetV3)](#7-image-based-cytology--classification-pipeline-yolov8--mobilenetv3)
8. [CBC Tabular Pipeline & Circular Target Leakage Audit](#8-cbc-tabular-pipeline--circular-target-leakage-audit)
9. [Default Stacking Ensemble Fusion vs. Experimental GMU Fusion](#9-default-stacking-ensemble-fusion-vs-experimental-gmu-fusion)
10. [Experimental GMU Implementation & Synthetic Training Protocol](#10-experimental-gmu-implementation--synthetic-training-protocol)
11. [Consolidated Metric Reconciliation & Old-vs-New Comparison](#11-consolidated-metric-reconciliation--old-vs-new-comparison)
12. [Backend Architecture & API Specifications (FastAPI)](#12-backend-architecture--api-specifications-fastapi)
13. [Frontend Architectures & Clinical Reporting (Streamlit, React & FPDF2)](#13-frontend-architectures--clinical-reporting-streamlit-react--fpdf2)
14. [Testing, Quality Assurance & Verification History](#14-testing-quality-assurance--verification-history)
15. [Current System State & Artifact Traceability](#15-current-system-state--artifact-traceability)
16. [Threats to Validity, Failure Modes & Engineering Limitations](#16-threats-to-validity-failure-modes--engineering-limitations)
17. [Future Clinical Roadmap & Research Priorities](#17-future-clinical-roadmap--research-priorities)
18. [Complete Reference & Artifact Inventory](#18-complete-reference--artifact-inventory)
19. [Evidence and Verification Status](#19-evidence-and-verification-status)

---

## 1. Project Overview, Motivation & Clinical Boundaries

### 1.1 The Clinical Problem

Anemia is a global public health condition affecting an estimated 1.92 billion individuals worldwide (~24.3% of the global population), with the highest morbidity observed among pregnant women, children under five years of age, and socioeconomically disadvantaged communities in low- and middle-income countries (LMICs). Clinically, anemia is defined as a reduction in circulating red blood cell (erythrocyte) mass or a subnormal concentration of intracellular hemoglobin, impairing the systemic oxygen-carrying capacity of blood.

Current clinical diagnosis relies on two primary modalities:
- **Automated Complete Blood Count (CBC):** Conducted using multi-channel automated hematology analyzers (flow cytometry and electrical impedance). While quantitative and rapid, automated analyzers cost $15,000–$80,000 USD, require expensive chemical reagents, need regular mechanical maintenance, and demand stable electrical infrastructure.
- **Peripheral Blood Smear (PBS) Microscopy:** Blood is spread across a glass slide, stained (Wright-Giemsa or Leishman), and visually examined under an optical light microscope at 400x to 1000x magnification. A pathologist manually enumerates cells and inspects erythrocyte cytomorphology (e.g., cell diameter, central pallor, shape irregularities). However, manual microscopy is slow (15–30 minutes per slide), labor-intensive, and prone to inter-observer variability exceeding 15–20% under high workloads.

### 1.2 Motivation for Multimodal AI

Conventional medical AI systems for hematology are almost universally unimodal:
- **Vision-Only Systems:** Classify blood smear crops into disease categories but cannot measure quantitative systemic hemoglobin concentration or cellular volumes.
- **Tabular-Only Systems:** Classify laboratory records but cannot inspect visual cellular morphology, sickle cells, target cells, or parasitological inclusions (such as *Plasmodium* malaria).

HemaVision AI was designed to explore a **dual-stream multimodal architecture** that processes both visual microscopy and numerical laboratory indices:
1. Enumerating Red Blood Cells (RBCs), White Blood Cells (WBCs), and Platelets directly from digital microscopic fields.
2. Estimating morphological anemia probability directly from smear photomicrographs via deep convolutional neural networks.
3. Classifying laboratory Complete Blood Count panels via tabular machine learning.
4. Fusing both modalities through decision-level ensemble meta-learning or deep feature-level gated neural units.

### 1.3 Intended Boundaries & Scope Delineation

To maintain absolute scientific and medical integrity, the system's operational scope is strictly delineated:
- **Preliminary Screening vs. Definitive Diagnosis:** HemaVision AI is a preliminary screening and decision-support prototype. It does not provide definitive diagnostic confirmation. In clinical pathology, definitive anemia diagnosis requires quantitative venous blood hemoglobin measurement interpreted within patient clinical context.
- **Academic Research Prototype:** All trained models, pipelines, and evaluation metrics are academic proofs-of-concept. The system must not be deployed in live clinical workflows without prospective clinical trials and regulatory clearance.

---

## 2. End-to-End System Architecture

HemaVision AI is engineered as an asynchronous, decoupled, multi-tier system. The complete end-to-end processing pipeline spans 13 sequential operational phases:

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       USER INTERACTION TIER                                           │
│       Streamlit Dashboard (Port 8501)          │        React 19 / TanStack SPA (Port 8080)           │
└────────────────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                                 │ HTTP Multipart POST /api/analyze
                                                 ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FASTAPI INFERENCE BACKEND (Port 8000)                                 │
│                                                                                                       │
│ 1. Request Ingestion & Multipart Parsing (`backend/routes/analyze.py`)                                │
│    - Parses optional smear image file & optional CBC numeric fields (Gender, Hb, MCV, MCH, MCHC)      │
│    - Dispatches mode: `default` (Stacking Ensemble) vs `experimental_gmu` (PyTorch GMU)               │
│                                                                                                       │
│    ┌───────────────────────────────────────────┴───────────────────────────────────────────┐          │
│    ▼                                                                                       ▼          │
│ ┌──────────────────────────────────────────────────────┐  ┌─────────────────────────────────────────┐ │
│ │              STREAM A: COMPUTER VISION               │  │        STREAM B: TABULAR CBC ML         │ │
│ │                                                      │  │                                         │ │
│ │ 2. Image Preprocessing (`src/preprocess.py`)         │  │ 5. CBC Validation (`src/anemia_model.py`)│ │
│ │    - Gaussian blur (3x3 kernel)                      │  │    - Range validation & Pydantic bounds │ │
│ │    - CLAHE on L-channel in LAB space                 │  │    - Gender parsing (0=Female, 1=Male)  │ │
│ │    - Standardization to 640x640                      │  │                                         │ │
│ │                                                      │  │ 6. CBC Feature Scaling & Inference      │ │
│ │ 3. Blood Cell Detection (`src/detect.py`)            │  │    - StandardScaler (`scaler.joblib`)   │ │
│ │    - YOLOv8n (`models/yolov8_blood/best.pt`)         │  │    - Random Forest (`anemia_model`)     │ │
│ │    - Classes: Platelet (0), RBC (1), WBC (2)         │  │    - Output: P(cbc), feature importance │ │
│ │    - Tiled inference fallback for high-res images    │  └────────────────────┬────────────────────┘ │
│ │                                                      │                       │                      │
│ │ 4. Smear Anemia Classifier (`anemia_image_model.py`) │                       │                      │
│ │    - MobileNetV3-Small (`mobilenetv3_anemia...pth`)  │                       │                      │
│ │    - Preprocessing: 224x224 RGB, ImageNet mean/std   │                       │                      │
│ │    - Output: P(img), confidence, entropy             │                       │                      │
│ └──────────────────────────┬───────────────────────────┘                       │                      │
│                            │                                                   │                      │
│                            └───────────────────────────┬───────────────────────┘                      │
│                                                        ▼                                              │
│ ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│ │                       MULTIMODAL FUSION LAYER (`src/anemia_fusion.py`)                            │ │
│ │                                                                                                   │ │
│ │ 7. Default Mode: Production Stacking Ensemble                                                     │ │
│ │    - 12-dimensional multimodal feature vector                                                     │ │
│ │    - Level-0: HistGradientBoosting + ExtraTrees + MLP                                             │ │
│ │    - Level-1: Calibrated LogisticRegression Meta-Learner (`fusion_model.joblib`)                  │ │
│ │    - Fallbacks: Average Fallback (50/50), Image-Only, CBC-Only, Insufficient Data                  │ │
│ │                                                                                                   │ │
│ │ 8. Selectable Mode: Experimental Gated Multimodal Unit (GMU)                                      │ │
│ │    - Feature extractors: 576-d MobileNetV3 visual embedding + 128-d CBC MLP embedding            │ │
│ │    - Candidate states: h_img, h_cbc in (-1, 1)^128 via tanh                                       │ │
│ │    - Dynamic Sigmoid Gate: z in (0, 1)^128                                                        │ │
│ │    - Fused latent vector: h_fused = z ⊙ h_img + (1 - z) ⊙ h_cbc                                   │ │
│ │    - Explicit Missing-Modality Masks: (m_img, m_cbc) clamp gates to 1.0 or 0.0                    │ │
│ └──────────────────────────────────────────────────────┬────────────────────────────────────────────┘ │
│                                                        │                                              │
│                   ┌────────────────────────────────────┴────────────────────────────────────┐         │
│                   ▼                                                                         ▼         │
│ ┌──────────────────────────────────────────────────┐ ┌──────────────────────────────────────────────┐ │
│ │         DATABASE PERSISTENCE TIER                │ │         CLINICAL PDF GENERATION              │ │
│ │        (`database/crud.py`, SQLite)              │ │        (`src/report_generator.py`)          │ │
│ │                                                  │ │                                              │ │
│ │ 9. Database Insertion (`blood_analyzer.db`)      │ │ 10. Asynchronous PDF Task (FPDF2)            │ │
│ │    - Unique GUID: BRA-YYYYMMDDHHMMSS-XXXXXX      │ │     - Document control header & patient table│ │
│ │    - Cell counts (RBC, WBC, Platelets)           │ │     - Annotated smear image thumbnail        │ │
│ │    - WBC differential percentages                │ │     - WBC differential pie chart (Matplotlib)│ │
│ │    - P(img), P(cbc), fused verdict, probability  │ │     - Multimodal risk gauge & strategy badge │ │
│ │    - Raw CBC inputs & active fusion strategy     │ │     - Mandatory clinical disclaimer banner   │ │
│ └──────────────────────────────────────────────────┘ └──────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Complete Repository & Code Walkthrough

This section provides a file-by-file technical audit of the local repository (`C:\Users\HP\Desktop\Blood_report_analyzer-main`), tracing internal logic, dependencies, and operational status.

### 3.1 Root Directory Architecture

```
Blood_report_analyzer-main/
├── backend/                  # FastAPI backend service routes, schemas, and entry point
├── database/                 # SQLAlchemy SQLite models, connection, and CRUD operations
├── data/                     # Raw, processed, and auxiliary datasets; SQLite database file
├── docs/                     # Technical markdown design documents and architectural audits
├── frontend/                 # Streamlit clinical dashboard application (`app.py`)
├── models/                   # Serialized model weights, scalers, and metric metadata
├── reports/                  # Generated clinical PDFs, audit reports, and master documentation
├── scripts/                  # Training pipelines, dataset preparation, and verification scripts
├── src/                      # Core computational modules (detection, classification, fusion)
├── tests/                    # Unit, integration, and architecture test suites
├── ui/                       # Modern React 19 / Vite / TanStack web frontend
├── requirements.txt          # Production Python dependency manifest
└── README.md                 # Project introduction and execution instructions
```

---

### 3.2 Core Inference Modules (`src/`)

#### 1. `src/preprocess.py` (Image Preprocessing Pipeline)
- **Purpose:** Preprocesses raw digital microscope photographs to standardize optical contrast, reduce high-frequency camera noise, and resize for downstream models.
- **Input:** Raw PIL Image or OpenCV BGR/RGB array.
- **Internal Logic:**
  1. Validates image dimensions and channels.
  2. Applies a $3 \times 3$ Gaussian smoothing kernel to attenuate high-frequency CMOS sensor noise.
  3. Converts image from BGR/RGB to CIE LAB color space.
  4. Extracts the luminance ($L^*$) channel and applies Contrast Limited Adaptive Histogram Equalization (CLAHE) with `clipLimit=2.0` and `tileGridSize=(8, 8)`. This amplifies local contrast around cellular boundaries without saturating bright erythrocyte central pallor.
  5. Merges channels back and converts to RGB/BGR format.
  6. Resizes to standard dimension $640 \times 640$ pixels for YOLOv8.
- **Dependencies:** `opencv-python` (`cv2`), `numpy`, `PIL.Image`.
- **Status:** **Verified Production.** Active in all API and Streamlit requests.

#### 2. `src/detect.py` (YOLOv8 Object Detection Pipeline)
- **Purpose:** Detects, classifies, and bounds blood cells (RBCs, WBCs, Platelets) in peripheral blood smears.
- **Input:** PIL Image or NumPy array.
- **Internal Logic:**
  - Loads YOLOv8n weights from [`models/yolov8_blood/best.pt`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/models/yolov8_blood/best.pt) (cached in memory as `_cached_model`).
  - Class mapping strictly enforced: `0: Platelet`, `1: RBC`, `2: WBC`.
  - **RGB vs. BGR Channel Inversion Guard:** Inspects input format. PIL Images are passed directly to `model.predict()` (which expects RGB natively). Raw NumPy arrays are converted via `cv2.cvtColor(img, cv2.COLOR_RGB2BGR)` prior to inference. This prevents the channel-swap bug that previously misclassified red erythrocytes as blue leukocytes.
  - **Tiled Inference Engine (`_detect_cells_tiled`):** Triggered when the image resolution exceeds $640 \times 640$ or when initial full-frame inference yields zero RBC detections. Partitions the smear into overlapping $320 \times 320$ patches with a 20% spatial overlap stride. Bounding boxes are projected back to global pixel coordinates, and global Non-Maximum Suppression (NMS) with an IoU threshold of 0.45 eliminates duplicate detections across patch seams.
- **Output:** Structured dictionary containing absolute cell counts (`rbc`, `wbc`, `platelet`), annotated bounding-box image, and bounding-box coordinates for leukocyte crops.
- **Dependencies:** `ultralytics.YOLO`, `cv2`, `torch`, `PIL.Image`.
- **Status:** **Verified Production.** Passes unit tests in `tests/test_cell_detection.py`.

#### 3. `src/classify_wbc.py` (WBC Differential Module)
- **Purpose:** Segments bounding-box crops of detected leukocytes and classifies them into the 5 mature functional subtypes: Neutrophil, Lymphocyte, Monocyte, Eosinophil, Basophil.
- **Internal Logic:**
  - Extracts square image patches centered on WBC bounding boxes.
  - Attempts to load trained classifier weights from `models/resnet50_wbc/wbc_classifier_sklearn.pth`.
  - **Fallback Reality:** The file `wbc_classifier_sklearn.pth` is **not present on disk** in this repository. When `Path.exists()` evaluates to `False`, the function logs a warning and transparently activates `_fallback_differential()`.
  - Distributes detected leukocytes according to WHO adult physiological reference ratios: Neutrophils 55%, Lymphocytes 30%, Monocytes 7%, Eosinophils 5%, Basophils 1%.
  - Sets strategy string to `"clinical_reference_fallback"`.
- **Output:** Dictionary containing count and percentage for each of the 5 leukocyte subtypes.
- **Status:** **Partially Implemented / Fallback Active.** Fully crash-safe; documents fallback behavior in all UI/PDF outputs.

#### 4. `src/anemia_image_model.py` (MobileNetV3 Smear Classifier)
- **Purpose:** Evaluates visual erythrocyte cytomorphology (anisocytosis, poikilocytosis, hypochromic central pallor) from blood smear photographs to output an image-based anemia probability.
- **Input:** PIL Image.
- **Internal Logic:**
  - Loads MobileNetV3-Small weights from [`models/anemia_image_cluster_split/mobilenetv3_anemia_cluster_split.pth`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/models/anemia_image_cluster_split/mobilenetv3_anemia_cluster_split.pth) (falling back to `models/anemia_image/mobilenetv3_anemia.pth` if unavailable).
  - Preprocesses input: Resize to $224 \times 224$, conversion to tensor, normalization via ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).
  - Forward pass through MobileNetV3-Small feature backbone and linear classification head.
  - Softmax normalization generates probabilities: `[P(Normal), P(Anemic)]`.
  - Decision threshold: Evaluated against calibrated Youden threshold $\tau^* = 0.5007$.
- **Output:** `ImageAnemiaPrediction` TypedDict containing `prediction` (`"Anemic"` vs `"Normal"`), `probability` $P(\text{img}) \in [0.0, 1.0]$, `confidence`, and `model_version`.
- **Status:** **Verified Production.** Evaluated across 811 atomic clusters.

#### 5. `src/anemia_model.py` (CBC Random Forest Classifier)
- **Purpose:** Evaluates tabular laboratory Complete Blood Count parameters to predict anemia risk.
- **Input:** Dictionary of CBC parameters (`Gender`, `Hemoglobin`, `MCV`, `MCH`, `MCHC`).
- **Internal Logic:**
  - Validates numerical types and checks values against physiological plausibility boundaries ($\text{Hb} \in [1.0, 25.0]\ \text{g/dL}, \text{MCV} \in [30.0, 150.0]\ \text{fL}$).
  - Formats feature vector `[Gender, Hemoglobin, MCH, MCHC, MCV]`.
  - Normalizes features using pre-fitted `StandardScaler` loaded from [`models/anemia/scaler.joblib`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/models/anemia/scaler.joblib).
  - Runs inference using serialized Random Forest classifier (`models/anemia/anemia_model.joblib`).
  - Extracts class probability $P(\text{cbc}) = P(y=1 \mid x)$ via `predict_proba`.
  - Attaches tree Gini feature importances to output payload.
- **Status:** **Verified Production.** Documents target leakage mechanism transparently.

#### 6. `src/anemia_fusion.py` (Multimodal Fusion Engine)
- **Purpose:** Central fusion orchestration layer. Merges visual smear features and laboratory CBC indices into a unified screening assessment.
- **Internal Logic:**
  - **Mode Selection:**
    - If `prediction_mode == "experimental_gmu"`, dispatches to `predict_gmu_experimental()`.
    - Otherwise, executes default `predict_fusion()`.
  - **`predict_fusion()` Logic:**
    1. Attempts to load [`models/anemia_fusion/fusion_model.joblib`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/models/anemia_fusion/fusion_model.joblib).
    2. Constructs a 12-dimensional multimodal feature vector:
       - Image stream: $P(\text{img})$, Confidence, Entropy.
       - CBC stream: $P(\text{cbc})$, Hb, MCV, MCH, MCHC, Gender.
       - Interaction terms: $\text{MCV}/\text{Hb}$ ratio, Discordance $|P_{\text{img}} - P_{\text{cbc}}|$, Synergy $P_{\text{img}} \times P_{\text{cbc}}$.
    3. Runs Level-0 base estimators (`HistGradientBoostingClassifier`, `ExtraTreesClassifier`, `MLPClassifier`) and Level-1 meta-learner (`LogisticRegression`).
    4. Strategy string returned: `"learned_stacking_ensemble"`.
    5. **Defensive Fallback Hierarchy:**
       - If model is missing but both inputs exist: `fused_prob = (P_img + P_cbc) / 2.0` (`strategy="average_fallback"`).
       - If only image exists: `fused_prob = P_img` (`strategy="image_only"`).
       - If only CBC exists: `fused_prob = P_cbc` (`strategy="cbc_only"`).
       - If neither exists: `status="insufficient_data"`, `fusion_strategy="none"`.
  - **`predict_gmu_experimental()` Logic:**
    1. Loads trained PyTorch GMU checkpoint (`gmu_fusion_experimental.pth`) and scaler (`gmu_experimental_scaler.joblib`).
    2. Pre-extracts 576-d visual embedding from MobileNetV3 backbone.
    3. Scales 5-d CBC vector.
    4. Evaluates dynamic gating $z \in (0, 1)^{128}$ and missing-modality masks $(m_{\text{img}}, m_{\text{cbc}})$.
    5. Returns prediction, probability, gate scalar summary, and synthetic limitation notices.
- **Status:** **Verified Production.** Dispatches both default stacking and experimental GMU paths.

#### 7. `src/models/gmu_fusion.py` (PyTorch GMU Deep Architecture)
- **Purpose:** Defines the deep learning `GMUAnemiaClassifier` neural network module in PyTorch.
- **Architecture:**
  - `image_proj`: `Linear(576, 128)` projecting pooled visual features to latent dimension $d=128$.
  - `cbc_mlp`: 2-layer MLP (`Linear(5, 64) -> LayerNorm(64) -> Swish -> Dropout(0.2) -> Linear(64, 128) -> LayerNorm(128) -> Swish`).
  - `gate_layer`: `Linear(256, 128)` computing gate pre-activations from concatenated candidate states $[h_{\text{img}}, h_{\text{cbc}}]$.
  - `classifier`: 2-layer classification head (`Linear(128, 64) -> LayerNorm(64) -> Swish -> Dropout(0.3) -> Linear(64, 1)`).
  - Explicit missing-modality mask enforcement in `forward()` clamps gates to 1.0 or 0.0 without numerical exceptions.
- **Status:** **Verified Production.** Passes 17 architectural unit tests in `tests/test_gmu_architecture.py`.

#### 8. `src/normal_ranges.py` (Clinical Reference Evaluation)
- **Purpose:** Converts raw cell counts into estimated high-power field (HPF) cellular densities and checks for clinical abnormalities against established hematological ranges.
- **Ranges Checked:** RBC density (15–35 / HPF), WBC density (1–4 / HPF), Platelet density (2–10 / HPF).
- **Status:** **Verified Production.**

#### 9. `src/report_generator.py` (Clinical PDF Report Engine)
- **Purpose:** Programmatically compiles comprehensive, publication-quality clinical PDF reports using `fpdf2`.
- **Layout & Structure:** Document metadata, patient laboratory CBC table with out-of-range directional flags, high-resolution annotated smear thumbnail, cell enumeration table, Matplotlib WBC differential pie chart, multimodal assessment card with active strategy badge, and persistent red-bordered clinical disclaimer banner.
- **Status:** **Verified Production.**

---

### 3.3 Backend API Service (`backend/`)

#### 1. `backend/main.py`
- Initializes FastAPI application instance (`title="Blood Report Analyzer API"`).
- Configures CORS middleware (`allow_origins=["*"]`, `allow_credentials=True`, `allow_methods=["*"]`, `allow_headers=["*"]`).
- Mounts static file directories (`/static` -> `data/`, `/reports` -> `reports/`).
- Registers route routers: `analyze_router`, `history_router`, `reports_router`.
- Serves root health check (`GET /` and `GET /api/health`).

#### 2. `backend/routes/analyze.py`
- Main analysis route: `POST /api/analyze`.
- Accepts `multipart/form-data`:
  - `file: UploadFile = File(None)`: Blood smear microscope photograph.
  - `gender: str = Form(None)`: Biological sex (`"Male"`, `"Female"`, `"0"`, `"1"`).
  - `hemoglobin: float = Form(None)`: Venous hemoglobin in g/dL.
  - `mcv: float = Form(None)`: Mean Corpuscular Volume in fL.
  - `mch: float = Form(None)`: Mean Corpuscular Hemoglobin in pg.
  - `mchc: float = Form(None)`: Mean Corpuscular Hemoglobin Concentration in g/dL.
  - `prediction_mode: str = Form("default")`: `"default"` (Stacking Ensemble) vs `"experimental_gmu"` (GMU).
- Orchestrates preprocessing, YOLOv8 detection, WBC differential, MobileNetV3 classification, CBC Random Forest, and fusion dispatch.
- Persists results to SQLite via `database/crud.py:create_analysis_record()`.
- Dispatches asynchronous PDF compilation via FastAPI `BackgroundTasks`.
- Returns complete `AnalysisResponse` JSON payload.

#### 3. `backend/schemas.py`
- Pydantic models enforcing strict request and response schema validation:
  - `CellCounts`: Integer counts (`rbc`, `wbc`, `platelet`, `total`).
  - `WBCDifferential`: Percentage distribution dictionary.
  - `NormalRanges`: Reference range evaluation flags.
  - `ImageAnemiaResult`: `prediction`, `probability`, `confidence`, `status`.
  - `CBCAnemiaResult`: `prediction`, `probability`, `feature_importances`.
  - `FusionResult`: `prediction`, `probability`, `strategy`, `model_version`.
  - `AnalysisResponse`: Complete unified response schema.

---

### 3.4 Database Persistence Layer (`database/`)

#### 1. `database/connection.py`
- Configures SQLAlchemy engine connected to local SQLite database: `sqlite:///data/blood_analyzer.db`.
- Sets `connect_args={"check_same_thread": False}` for multi-threaded asynchronous FastAPI compatibility.
- Factory `get_db()` provides transactional database sessions via dependency injection.

#### 2. `database/models.py`
- Defines `Analysis` ORM model mapped to the `analyses` table in SQLite:
  - `id`: Primary key integer.
  - `analysis_id`: Unique GUID string (`BRA-YYYYMMDDHHMMSS-XXXXXX`).
  - `timestamp`: ISO-8601 UTC execution timestamp.
  - `image_path` / `annotated_image_path`: Relative paths to stored images.
  - `rbc_count`, `wbc_count`, `platelet_count`: Integer cell counts.
  - `wbc_differential_json`: Serialized JSON string of leukocyte differential.
  - `image_anemia_prediction`, `image_anemia_probability`: Smear classifier outputs.
  - `cbc_anemia_prediction`, `cbc_anemia_probability`: CBC classifier outputs.
  - `cbc_inputs_json`: Serialized JSON string of raw user CBC parameters.
  - `final_prediction`, `final_probability`: Final fused diagnostic verdict.
  - `fusion_strategy`: Active routing strategy string.
  - `pdf_report_path`: File path to generated PDF.

#### 3. `database/crud.py`
- Implements transactional CRUD helper functions:
  - `create_analysis_record()`: Inserts newly completed analysis records into SQLite.
  - `get_analysis_by_id()`: Queries single analysis record by GUID string.
  - `get_recent_analyses()`: Queries paginated history records sorted in reverse chronological order.

---

### 3.5 Frontend Implementations

#### 1. Streamlit Clinical Dashboard (`frontend/app.py`)
- Python-native clinical interface running on port 8501.
- Features dual-column layout for simultaneous drag-and-drop microscope smear upload and CBC numeric parameter input.
- Sidebar radio selector enables switching between Production Stacking Ensemble and Experimental GMU Fusion, dynamically updating warning banners.
- Displays color-coded YOLOv8 bounding boxes, cell count metric cards, WBC differential charts, and side-by-side probability gauges.
- One-click button triggers asynchronous PDF download.

#### 2. React 19 / TanStack Web Application (`ui/`)
- Enterprise Single Page Application (SPA) built with React 19, TypeScript, Vite 8, TanStack Router, TanStack Query, and Tailwind CSS 4.
- High-contrast slate/navy diagnostic dashboard (`#0f172a` to `#1e1b4b`) engineered for presentation projectors and laboratory monitors.
- Located in `ui/` directory with independent `package.json` manifest.

---

## 4. Dataset History & Data Integrity Audit

### 4.1 Dataset Inventory & Sources

```
                                    DATASET PROVENANCE HIERARCHY
┌───────────────────────────┬───────────────────────────┬───────────────────────────┬───────────────────────────┐
│       BCCD Smear Data     │       AneRBC Smear Data   │     Chula-RBC-12 Smears   │     Master CBC Dataset    │
├───────────────────────────┼───────────────────────────┼───────────────────────────┼───────────────────────────┤
│ - Origin: Shenggan (MIT)  │ - Origin: Mendeley Data   │ - Origin: Chulalongkorn U │ - Origin: Kaggle + v4     │
│ - Type: Smear Bounding Box│ - Type: Binary Microscopy │ - Type: 12-Class Diseases │ - Type: Tabular Laboratory│
│ - Images: 364 raw         │ - Images: 738 raw         │ - Images: 121 extracted   │ - Records: 2,695 total    │
│ - Cells: 4,888 instances  │ - Classes: Anemic/Normal  │ - Target: RBC Diseases    │ - Features: Hb, MCV, MCH..│
└───────────────────────────┴───────────────────────────┴───────────────────────────┴───────────────────────────┘
```

1. **BCCD Dataset (Blood Cell Count and Detection):**
   - Source: Open-source repository curated by Shenggan (MIT License).
   - Content: 364 annotated peripheral blood smear images containing 4,888 cell instances (4,155 RBCs, 372 WBCs, 361 Platelets).
   - Augmentation: Training platelet images received 3x geometric augmentations (horizontal flip, vertical flip, composite rotation) expanding the training split to 674 images to mitigate platelet class imbalance.
2. **AneRBC Dataset:**
   - Source: Mendeley Data (DOI: 10.17632/hms3sjzt7f.1).
   - Content: Peripheral blood smear photomicrographs categorized into Anemic and Normal blood smears.
3. **Chula-RBC-12 Dataset:**
   - Source: Biomedical Informatics Laboratory, Chulalongkorn University.
   - Content: 121 specialized erythrocyte disease photomicrographs extracted from `chula_main.zip` into `data/anerbc_raw/Anemia/` (covering Hereditary Spherocytosis, Hereditary Elliptocytosis, Iron Deficiency, Megaloblastic Anemia, and Thalassemia).
4. **Master Complete Blood Count Dataset (`master_cbc_dataset.csv`):**
   - Source: Ingestion and harmonization of the open-access Kaggle Anemia Dataset (1,421 records) and clinical complete blood count records (`diagnosed_cbc_data_v4.csv`, 1,281 records) into [`data/anemia/processed/master_cbc_dataset.csv`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/data/anemia/processed/master_cbc_dataset.csv).
   - Total Records: 2,695 patient records (1,429 Anemic cases [53.0%], 1,266 Normal cases [47.0%]).
   - Features: `Gender, Hemoglobin, MCV, MCH, MCHC, Result, Diagnosis, Source`.

---

### 4.2 Near-Duplicate Audit & Review of the 27 Candidate Pairs

To identify potential data contamination, an automated perceptual hashing (dHash) and deep feature cosine similarity audit was executed across all 837 microscopic blood smear images.

#### Audit Summary:
- **Candidate Pairs Identified (dHash Hamming distance $\le 4$):** Exactly **27 candidate pairs**.
- **Exact Duplicates (Pixel MSE $< 10^{-4}$):** **25 pairs** (e.g., `725.jpg` and `chula_disease_sampleA0053.JPG`).
- **Likely Same-Slide Frames (dHash $\le 1$, Cosine Sim $\ge 0.985$, Same Class):** **2 pairs** (`251.jpg` vs `252.jpg`, `441.jpg` vs `440.jpg`).
- **Cross-Split Contamination in Unclustered Baseline:** **14 pairs** crossed training, validation, and test boundaries in the unclustered split.
- **Uncertain / Cross-Class Duplicate Pair:** Exactly **1 pair** (`151.jpg` [Anemic] vs `152.jpg` [Normal]). Their dHash distance is 4, cosine similarity is 0.9961, and pixel MSE is 10.41 ($0.00016$ normalized). Because their clinical labels conflict, they were intentionally **not** merged into a single cluster to preserve class purity; they remain separate singletons.

#### Complete Inventory of the 27 Near-Duplicate Candidate Pairs:

| Pair ID | Image 1 | Split 1 | Image 2 | Split 2 | dHash Dist | Cosine Sim | Classification Category | Cross-Split in Old Baseline? |
|:---:|---|:---:|---|:---:|:---:|:---:|---|:---:|
| 1 | `151.jpg` (Anemic) | `train` | `152.jpg` (Normal) | `train` | 4 | 0.9961 | `EXACT_DUPLICATE` (Conflicting Label) | No |
| 2 | `251.jpg` (Anemic) | `train` | `252.jpg` (Anemic) | `test` | 3 | 0.9978 | `LIKELY_SAME_SLIDE_FRAME` | ⚠️ **YES** |
| 3 | `262.jpg` (Anemic) | `train` | `263.jpg` (Anemic) | `val` | 0 | 0.9945 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 4 | `267.jpg` (Anemic) | `train` | `273.jpg` (Anemic) | `train` | 0 | 0.9932 | `EXACT_DUPLICATE` | No |
| 5 | `268.jpg` (Anemic) | `train` | `274.jpg` (Anemic) | `train` | 1 | 0.9985 | `EXACT_DUPLICATE` | No |
| 6 | `269.jpg` (Anemic) | `train` | `275.jpg` (Anemic) | `test` | 0 | 0.9993 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 7 | `270.jpg` (Anemic) | `train` | `276.jpg` (Anemic) | `train` | 0 | 0.9995 | `EXACT_DUPLICATE` | No |
| 8 | `271.jpg` (Anemic) | `train` | `277.jpg` (Anemic) | `val` | 1 | 0.9992 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 9 | `272.jpg` (Anemic) | `train` | `278.jpg` (Anemic) | `train` | 0 | 0.9993 | `EXACT_DUPLICATE` | No |
| 10 | `725.jpg` (Anemic) | `train` | `chula_disease_sampleA0053.JPG` (Anemic) | `train` | 0 | 0.9999 | `EXACT_DUPLICATE` | No |
| 11 | `726.jpg` (Anemic) | `train` | `chula_disease_sampleA0052.JPG` (Anemic) | `train` | 0 | 0.9998 | `EXACT_DUPLICATE` | No |
| 12 | `727.jpg` (Anemic) | `train` | `chula_disease_sampleA0051.JPG` (Anemic) | `train` | 0 | 0.9998 | `EXACT_DUPLICATE` | No |
| 13 | `728.jpg` (Anemic) | `train` | `chula_disease_sampleA0050.JPG` (Anemic) | `train` | 1 | 0.9999 | `EXACT_DUPLICATE` | No |
| 14 | `729.jpg` (Anemic) | `train` | `chula_disease_sampleA0049.JPG` (Anemic) | `train` | 0 | 0.9999 | `EXACT_DUPLICATE` | No |
| 15 | `731.jpg` (Anemic) | `train` | `chula_disease_sampleA0047.JPG` (Anemic) | `test` | 0 | 0.9999 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 16 | `732.jpg` (Anemic) | `train` | `chula_disease_sampleA0046.JPG` (Anemic) | `val` | 0 | 0.9999 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 17 | `736.jpg` (Anemic) | `train` | `chula_disease_sampleA0042.JPG` (Anemic) | `test` | 0 | 0.9998 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 18 | `737.jpg` (Anemic) | `train` | `chula_disease_sampleA0041.JPG` (Anemic) | `test` | 0 | 0.9998 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 19 | `738.jpg` (Anemic) | `train` | `chula_disease_sampleA0039.JPG` (Anemic) | `train` | 1 | 0.9999 | `EXACT_DUPLICATE` | No |
| 20 | `chula_disease_sampleA0044.JPG` (Anemic) | `train` | `734.jpg` (Anemic) | `test` | 0 | 0.9998 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 21 | `chula_disease_sampleA0045.JPG` (Anemic) | `train` | `733.jpg` (Anemic) | `test` | 3 | 0.9999 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 22 | `chula_disease_sampleA0048.JPG` (Anemic) | `train` | `730.jpg` (Anemic) | `val` | 1 | 0.9998 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 23 | `383.jpg` (Normal) | `train` | `384.jpg` (Normal) | `train` | 0 | 0.9979 | `EXACT_DUPLICATE` | No |
| 24 | `441.jpg` (Normal) | `train` | `440.jpg` (Normal) | `val` | 0 | 0.9996 | `LIKELY_SAME_SLIDE_FRAME` | ⚠️ **YES** |
| 25 | `233.jpg` (Anemic) | `val` | `234.jpg` (Anemic) | `test` | 1 | 0.9985 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 26 | `243.jpg` (Anemic) | `val` | `244.jpg` (Anemic) | `test` | 0 | 0.9978 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 27 | `735.jpg` (Anemic) | `val` | `chula_disease_sampleA0043.JPG` (Anemic) | `val` | 0 | 0.9998 | `EXACT_DUPLICATE` | No |

---

### 4.3 Atomic Cluster Construction & Partitioning

To resolve the 14 cross-split duplicate pairs, connected component clustering merged all identical and near-duplicate frames into atomic, indivisible cluster units:
- **Total Images Processed:** 837 images.
- **Atomic Clusters Constructed:** Exactly **811 atomic clusters**.
- **Split Strategy:** Stratified group splitting allocating entire clusters to Train, Validation, or Test:

| Split Partition | Atomic Clusters | Total Images | Class 0 (Normal) | Class 1 (Anemic) | Cross-Split Cluster Overlap |
|---|:---:|:---:|:---:|:---:|:---:|
| **Training Split** | 567 | 583 | 86 | 497 | **0 clusters** |
| **Validation Split** | 121 | 125 | 19 | 106 | **0 clusters** |
| **Held-Out Test Split** | 123 | 129 | 19 | 110 | **0 clusters** |
| **Complete Dataset** | **811** | **837** | **124** | **713** | **0 clusters** |

$$\text{Sum Check:} \quad 567 + 121 + 123 = 811 \quad (100.0\% \text{ reconciled})$$

> [!IMPORTANT]
> **Cluster-Level vs. Patient-Level Independence:** The AneRBC and Chula-RBC-12 datasets do not provide clinical patient identifiers. While clustering eliminates field-of-view slide leakage, **it does not guarantee patient-level independence**. Multiple slides may still originate from the same patient without our knowledge.

---

## 5. Original Implementation & Baseline Results

### 5.1 Earliest System Architecture

The earliest recorded implementation of HemaVision AI (preserved in `models/anemia_image/mobilenetv3_anemia_original_baseline.pth` and `models/anemia/anemia_model.joblib`) consisted of:
1. **YOLOv8 Detection:** Early model trained on BCCD with an undetected RGB vs BGR channel-order bug.
2. **MobileNetV3 Image Model:** Fine-tuned on raw AneRBC images with unweighted cross-entropy loss under a default 0.5 decision threshold.
3. **CBC Random Forest:** Trained on the Kaggle Anemia dataset using all 5 features including `Hemoglobin`.
4. **Multimodal Fusion:** Simple probability averaging ($P_{\text{fused}} = (P_{\text{img}} + P_{\text{cbc}}) / 2$) or unweighted logistic regression.

### 5.2 Baseline Performance & Failure Modes

#### Failure Mode 1: Image Specificity Collapse (5.0%)
- **Historical Result:** Accuracy = 80.56%, Sensitivity = 97.73%, **Specificity = 5.00%**, F1 = 0.8912, ROC-AUC = 0.7477 (evaluated on 108 unpurged test images: 88 Anemic, 20 Normal).
- **Confusion Matrix:** $\text{TN} = 1, \text{FP} = 19, \text{FN} = 2, \text{TP} = 86$.
- **Cause:** Severe class imbalance (5.7:1 ratio) in training data caused unweighted cross-entropy gradients to overwhelmingly favor the positive class. The model classified 19 out of 20 healthy images as anemic.

#### Failure Mode 2: Circular Target Leakage in CBC Model (100.0%)
- **Historical Result:** Accuracy = 100.0%, Sensitivity = 100.0%, Specificity = 100.0%, ROC-AUC = 1.0000 (evaluated on 214 test samples from Kaggle dataset).
- **Confusion Matrix:** $\text{TN} = 121, \text{FP} = 0, \text{FN} = 0, \text{TP} = 93$.
- **Cause:** The target label (`Result`) was derived directly from WHO Hemoglobin cutoffs. Supplying raw Hemoglobin as an input feature enabled the decision trees to simply memorize the cutoff threshold (Hemoglobin accounted for 83.32% of feature importance).

#### Failure Mode 3: YOLO RGB vs. BGR Channel Inversion
- **Symptom:** Blood smears with clear red erythrocytes reported zero or near-zero RBCs, while reporting dozens of false leukocytes.
- **Cause:** Passing PIL RGB arrays directly into OpenCV/YOLO functions that interpret arrays in BGR order inverted the red and blue channels, causing pink RBCs to appear deep purple and trigger leukocyte bounding boxes.

---

## 6. Chronological Development History & Major Modifications

The chronological evolution of HemaVision AI is reconstructed from file timestamps, script logs, and codebase modifications:

```
                                    SYSTEM DEVELOPMENT TIMELINE
  2026-08-22 ──► Milestone 1: Initial Repository & Baseline Pipelines (YOLOv8, RF CBC, Baseline Smear)
  2026-08-29 ──► Milestone 2: Dataset Expansion (Chula RBC Diseases, Master CBC Consolidation)
  2026-08-30 ──► Milestone 3: Channel Swap Bug Fix & Results Page UI Polish (RGB/BGR fix, contrast)
  2026-10-22 ──► Milestone 4: Multimodal Stacking Ensemble (v2.0.0, 12-d feature space, meta-learner)
  2026-10-23 ──► Milestone 5: Near-Duplicate Audit & Cluster-Level Split Remediation (811 clusters)
  2026-10-23 ──► Milestone 6: CBC Target Leakage Audit & Leakage-Aware Experiment (Hb-free screening)
  2026-10-23 ──► Milestone 7: Deep GMU Implementation & Experimental Training (PyTorch GMU, masks)
  2026-10-23 ──► Milestone 8: Full Verification, QA Suite Expansion & Master Documentation (42 tests)
```

---

## 7. Image-Based Cytology & Classification Pipeline (YOLOv8 & MobileNetV3)

### 7.1 YOLOv8n Blood Cell Detection Pipeline

- **Architecture:** Anchor-free single-stage detector; 73 layers, 3,006,233 parameters, 8.1 GFLOPs.
- **Pretrained Checkpoint:** `models/yolov8_blood/best.pt`.
- **Operational Thresholds:** Confidence threshold $\text{conf} = 0.25$, IoU threshold for NMS $\text{iou} = 0.45$. In low-density frames, confidence dynamically scales to $0.15$ to catch faint thrombocytes.
- **Class Mappings:** Strictly mapped as `0: Platelet`, `1: RBC`, `2: WBC`.
- **Tiled Inference Strategy:** Slices high-resolution smears ($>640 \times 640$) into $320 \times 320$ patches with a 20% spatial overlap stride, followed by global cross-tile NMS.

### 7.2 MobileNetV3-Small Classification Pipeline

- **Architecture:** Pretrained MobileNetV3-Small backbone with hard-swish activations and Squeeze-and-Excitation channel attention.
- **Input Pipeline:** Standardized $224 \times 224 \times 3$ RGB tensors normalized via ImageNet mean ($\mu = [0.485, 0.456, 0.406]$) and standard deviation ($\sigma = [0.229, 0.224, 0.225]$).
- **Latent Embedding:** Global average pooling produces a 576-dimensional visual feature vector.
- **Classification Head:** Linear layer projecting $576 \to 2$ logits (`Normal` vs `Anemic`), followed by softmax normalization.
- **Calibrated Operating Threshold:** Calibrated on the validation split via Youden's J statistic at $\tau^* = 0.5007$.

---

## 8. CBC Tabular Pipeline & Circular Target Leakage Audit

### 8.1 Clinical Rationale & Standard Architecture

The CBC tabular stream processes 5 standard laboratory measurements:
1. `Gender`: Biological sex (0 = Female, 1 = Male)
2. `Hemoglobin`: Venous hemoglobin concentration (g/dL)
3. `MCV`: Mean Corpuscular Volume (fL)
4. `MCH`: Mean Corpuscular Hemoglobin (pg)
5. `MCHC`: Mean Corpuscular Hemoglobin Concentration (g/dL)

Features are normalized using `StandardScaler` (`scaler.joblib`) and classified via Random Forest (`n_estimators=100`, `class_weight='balanced'`).

### 8.2 The Target Leakage Mechanism

In clinical hematology, anemia is defined by low venous hemoglobin (WHO criteria: $<12.0\ \text{g/dL}$ in females, $<13.0\ \text{g/dL}$ in males). In public machine learning datasets (Kaggle Anemia and clinical CBC tables), the target label (`Result`) was assigned mathematically by applying this exact threshold rule to the `Hemoglobin` and `Gender` columns:
$$y = \mathbb{I}\left( (\text{Gender} = 0 \land \text{Hb} < 12.0) \lor (\text{Gender} = 1 \land \text{Hb} < 13.0) \right)$$

When `Hemoglobin` is provided as an input feature:
1. Decision trees trivially learn the exact threshold cutoff, achieving **99.51% to 100.0% accuracy**.
2. Gini impurity analysis reveals that `Hemoglobin` alone accounts for **79.98% to 83.32%** of all decision splits.
3. The model memorizes a labeling threshold rule rather than biological disease patterns. In a clinical lab where hemoglobin is already measured, machine learning prediction of anemia is medically redundant.

### 8.3 The Leakage-Aware Screening Experiment (Without Hemoglobin)

To establish whether machine learning provides genuine utility when hemoglobin cannot be directly measured, an independent experiment was conducted on 809 held-out patients from the Master CBC Dataset (2,695 total records):
- **Features Evaluated:** `[Gender, MCV, MCH, MCHC]` (**strictly excluding `Hemoglobin`**).
- **Results:**
  - **Random Forest:** Accuracy = **83.56%**, Sensitivity = **89.28%**, Specificity = **77.11%**, F1 = **0.8521**, ROC-AUC = **0.8974**.
  - **Gradient Boosting:** Accuracy = 80.72%, Sensitivity = 83.92%, Specificity = 77.11%, ROC-AUC = 0.8774.
  - **Logistic Regression:** Accuracy = 64.52%, Sensitivity = 74.83%, Specificity = 52.89%, ROC-AUC = 0.6646.
- **Feature Weight Redistribution:** Without Hemoglobin, Random Forest redistributed its decision splits across red cell cytometric indices: **MCV (38.63%)**, **MCH (27.77%)**, **MCHC (24.08%)**, and **Gender (9.52%)**.
- **Biological Finding:** In iron deficiency and thalassemia, impaired hemoglobin synthesis forces microcytosis (low MCV) and hypochromia (low MCH/MCHC). Red cell morphology acts as a viable surrogate screening marker, but cannot detect normocytic anemias where cell volume is normal.

---

## 9. Default Stacking Ensemble Fusion vs. Experimental GMU Fusion

HemaVision AI integrates two distinct multimodal fusion paradigms:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    MULTIMODAL FUSION ARCHITECTURES                                     │
├────────────────────────────────────────────────────┬───────────────────────────────────────────────────┤
│    PRODUCTION ENGINE: Stacking Ensemble (v2.0.0)   │    EXPERIMENTAL ENGINE: Gated Multimodal Unit     │
├────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
│ - Paradigm: Decision-level + Meta-feature stacking │ - Paradigm: Feature-level deep neural gating      │
│ - Input: 12-dimensional engineered feature vector  │ - Input: 576-d image embedding + 5-d CBC vector   │
│ - Estimators: HistGB + ExtraTrees + MLP -> LogReg  │ - Architecture: Shared latent space (d=128)       │
│ - Features: Probabilities, entropy, ratios, discord│ - Mechanism: Dynamic elementwise sigmoid gate z   │
│ - Runtime: Active by default in API and UI         │ - Masking: Explicit binary masks (m_img, m_cbc)   │
│ - Fallback: Transparent average & single-modality  │ - Runtime: Selectable experimental prototype     │
└────────────────────────────────────────────────────┴───────────────────────────────────────────────────┘
```

### 9.1 Default Stacking Ensemble Engine

The production model ([`models/anemia_fusion/fusion_model.joblib`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/models/anemia_fusion/fusion_model.joblib)) implements a multi-paradigm Stacking Ensemble:
- **Level-0 Base Estimators:**
  1. `HistGradientBoostingClassifier`: 150 iterations, learning rate 0.05, max_depth 5.
  2. `ExtraTreesClassifier`: 100 randomized decision trees, max_depth 6.
  3. `MLPClassifier`: 64-32 architecture, ReLU activation, Adam optimizer.
- **Level-1 Meta-Learner:** Calibrated `LogisticRegression(class_weight='balanced')` trained via 5-fold cross-validation on out-of-fold probability vectors.
- **Evaluation Provenance (99.44% Accuracy):** Evaluated on 360 held-out samples from a 2,400-sample **synthetically constructed dataset** assembled by randomly pairing MobileNetV3 visual predictions from 859 smear images with CBC records from 2,695 patients (`scripts/train_anemia_fusion.py:182-250`). Demonstrates ensemble performance on synthetic tabular combinations, **not clinical patient accuracy**.

### 9.2 Defensive Runtime Fallback Priority Hierarchy

In [`src/anemia_fusion.py:89-240`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/anemia_fusion.py#L89-L240), `predict_fusion()` dynamically routes requests based on input completeness:
1. **Learned Stacking Ensemble (`strategy="learned_stacking_ensemble"`):** Active when both image and CBC inputs are valid and `fusion_model.joblib` is loaded.
2. **Probability Averaging Fallback (`strategy="average_fallback"`):** Active if model weights are unavailable or feature columns are incomplete:
   $$P_{\text{fused}} = \frac{P_{\text{img}} + P_{\text{cbc}}}{2}$$
3. **Single-Modality Image-Only (`strategy="image_only"`):** Active when only a smear photograph is provided ($P_{\text{fused}} = P_{\text{img}}$).
4. **Single-Modality CBC-Only (`strategy="cbc_only"`):** Active when only laboratory CBC values are provided ($P_{\text{fused}} = P_{\text{cbc}}$).
5. **Insufficient Data Guardrail (`status="insufficient_data"`):** If neither modality is provided, returns prediction `"Undetermined"` with probability $0.0$.

---

## 10. Experimental GMU Implementation & Synthetic Training Protocol

### 10.1 Mathematical Formulation of the GMU

The Gated Multimodal Unit (GMU) projects diverse modalities into a shared latent space and learns an internal gating vector to dynamically weight feature contributions:

```
Blood Smear Image (3 x 224 x 224)              Laboratory CBC Features (5-d)
            │                                                │
MobileNetV3 Feature Extractor                     Multi-Layer Perceptron (MLP)
  (576-d global pooled vector)                   (5 -> 64 -> LayerNorm -> 128)
            │                                                │
Linear Projection (576 -> 128)                               │
            │                                                │
            ▼                                                ▼
   Image Vector: x_img (128-d)                      CBC Vector: x_cbc (128-d)
            │                                                │
            ├───────────────────────┬────────────────────────┤
            │                       │                        │
            ▼                       ▼                        ▼
     Candidate State         Candidate State            Dynamic Gate
   h_img = tanh(W_i x_i)   h_cbc = tanh(W_c x_c)   z = σ(W_z [h_i, h_c] + b_z)
            │                       │                        │
            └───────────┬───────────┴────────────────────────┘
                        ▼
              Fused Latent Embedding:
          h_fused = z ⊙ h_img + (1 - z) ⊙ h_cbc
                        │
                        ▼
              Classification Head
      Linear(128 -> 64) -> Swish -> Dropout(0.3) -> Linear(64 -> 1)
                        │
                        ▼
               Sigmoid Activation
                        │
                        ▼
             Predicted Probability P(Anemia)
```

1. **Latent Projections:**
   $$x_{\text{img}} = W_{\text{proj}} f_{\text{img}} + b_{\text{proj}} \in \mathbb{R}^{d}$$
   $$x_{\text{cbc}} = \text{Swish}\left(\text{LN}\left(W_{\text{cbc}, 2} (\text{Swish}(W_{\text{cbc}, 1} f_{\text{cbc}} + b_1)) + b_2\right)\right) \in \mathbb{R}^{d}$$
   where latent dimension $d = 128$.
2. **Candidate Modality Embeddings:**
   $$h_{\text{img}} = \tanh(W_{\text{img}} x_{\text{img}} + b_{\text{img}}) \in (-1, 1)^{d}$$
   $$h_{\text{cbc}} = \tanh(W_{\text{cbc}} x_{\text{cbc}} + b_{\text{cbc}}) \in (-1, 1)^{d}$$
3. **Dynamic Multimodal Gating:**
   $$z = \sigma\left(W_z [h_{\text{img}}, h_{\text{cbc}}] + b_z\right) \in (0, 1)^{d}$$
4. **Gated Multimodal Combination:**
   $$h_{\text{fused}} = z \odot h_{\text{img}} + (\mathbf{1} - z) \odot h_{\text{cbc}}$$
5. **Classification Logit:**
   $$\hat{y} = \sigma\left( W_{\text{head}, 2} \left( \text{Dropout}\left(\text{Swish}\left(\text{LN}\left(W_{\text{head}, 1} h_{\text{fused}} + b_{h1}\right)\right)\right) \right) + b_{h2} \right)$$

### 10.2 Synthetic Training Protocol & Split Isolation

- **Training Script:** [`scripts/train_gmu_fusion.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/scripts/train_gmu_fusion.py).
- **Split Isolation:** Train pairs (2,000) sampled exclusively from the 567 training image clusters and training CBC rows; Validation pairs (400) sampled from validation clusters; Test pairs (400) sampled from held-out test clusters. No image cluster or CBC row crossed partition boundaries.
- **Discordance Protocol:** For discordant pairs (8% of data), target labels were assigned according to the WHO Hemoglobin threshold of the CBC component, reflecting clinical hematological practice where venous hemoglobin governs systemic diagnosis.
- **Optimization:** AdamW ($\eta = 0.001$, weight decay $10^{-4}$), `BCEWithLogitsLoss()`, 10 epochs.
- **Checkpoint Selection:** Epoch 9 achieved the minimum validation loss of **0.0617** (Acc: 98.50%) and was frozen as [`models/anemia_fusion/gmu_fusion_experimental.pth`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/models/anemia_fusion/gmu_fusion_experimental.pth).

---

## 11. Consolidated Metric Reconciliation & Old-vs-New Comparison

### 11.1 Master Old-vs-New Metric Comparison Tables

#### Table A: Image-Based Anemia Models (Smear Photomicrographs)

| Model Iteration | Checkpoint Artifact Path | Evaluation Split / Sample Count | Accuracy | Sensitivity | Specificity | F1-Score | ROC-AUC | Methodological Notes & Key Limitations |
|---|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Old Baseline (Unweighted)** | `mobilenetv3_anemia_original_baseline.pth` | Unpurged 108 test images (AneRBC) | 80.56% | 97.73% | **5.00%** | 0.8912 | 0.7477 | 5.7:1 class imbalance caused specificity collapse (19 FP, 1 TN). Unclustered split. |
| **Baseline Re-evaluation** | `mobilenetv3_anemia_original_baseline.pth` | Purged 128 test images (AneRBC) | 81.25% | 83.33% | 70.00% | 0.8824 | 0.8752 | Baseline weights evaluated on cleaned 128-image split with cross-class duplicate purged. |
| **Weighted Retraining** | `mobilenetv3_anemia.pth` | Purged 128 test images (AneRBC) | 75.78% | 75.93% | 75.00% | 0.8410 | 0.8500 | Retrained with inverse class weights. Value 0.8148 in draft table was a clerical error ($88/108$). |
| **Cluster-Remediated Model** | `mobilenetv3_anemia_cluster_split.pth` | Remediated 129 test images (811 clusters) | **75.19%** | **77.27%** | **63.16%** | **0.8416** | **0.8167** | **Active Production Smear Model.** 811 atomic clusters; 0 cross-split leakage. Calibrated at $\tau^* = 0.5007$. |

#### Table B: Complete Blood Count Models (Tabular Laboratory Records)

| Model Configuration | Feature Vector Used | Evaluation Split / Dataset | Accuracy | Sensitivity | Specificity | F1-Score | ROC-AUC | Methodological Reality & Clinical Assessment |
|---|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Standard Random Forest** | `[Gender, Hb, MCV, MCH, MCHC]` | Kaggle Test (214 records) | 100.0% | 100.0% | 100.0% | 1.0000 | 1.0000 | **Circular Target Leakage:** Target label derived from Hb; Hb feature importance is 83.32%. |
| **Master Dataset Random Forest** | `[Gender, Hb, MCV, MCH, MCHC]` | Master CBC Test (809 records) | 99.51% | 99.53% | 99.47% | 0.9953 | 0.9999 | Target leakage present on 2,695-record dataset. Demonstrates threshold-reproduction. |
| **Leakage-Aware Screening Model** | `[Gender, MCV, MCH, MCHC]` (**NO Hb**) | Master CBC Test (809 records) | **83.56%** | **89.28%** | **77.11%** | **0.8521** | **0.8974** | **Genuine Screening Model:** Strictly omits Hb. Proves MCV/MCH act as biological surrogate markers. |

#### Table C: Multimodal Fusion Models (Comparative Synthetic Benchmark)

| Fusion Engine | Checkpoint Artifact Path | Evaluation Split / Sample Count | Accuracy | Sensitivity | Specificity | F1-Score | ROC-AUC | Pairing Nature & Operational Status |
|---|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Production Stacking Ensemble** | `models/anemia_fusion/fusion_model.joblib` | Held-out Tabular (360 synthetic pairs) | **99.44%** | **99.44%** | **99.44%** | **0.9944** | **0.9999** | **Active Default Engine:** 12-d Stacking Ensemble. Evaluated on synthetic tabular combinations. |
| **Probability Averaging Baseline** | Mathematical fallback rule | Held-out Synthetic (400 synthetic pairs) | 97.75% | 98.00% | 97.50% | 0.9776 | 0.9922 | Baseline evaluated on identical 400 GMU test pairs. Simple average of probabilities. |
| **Experimental GMU Dual-Modality** | `models/anemia_fusion/gmu_fusion_experimental.pth` | Held-out Synthetic (400 synthetic pairs) | **97.50%** | **97.00%** | **98.00%** | **0.9749** | **0.9978** | **Selectable Experimental Prototype:** PyTorch GMU ($d=128$, $\bar{z}=0.4499$). Evaluated on synthetic pairs. |
| **GMU CBC-Only Mode** | Evaluated with mask $m_{\text{img}}=0, m_{\text{cbc}}=1$ | Held-out Synthetic (400 synthetic pairs) | 97.00% | 98.00% | 96.00% | 0.9703 | 0.9980 | Gate clamped to $z=\mathbf{0}$. Evaluated on identical 400 synthetic test pairs. |
| **GMU Image-Only Mode** | Evaluated with mask $m_{\text{img}}=1, m_{\text{cbc}}=0$ | Held-out Synthetic (400 synthetic pairs) | 69.00% | 71.00% | 67.00% | 0.6961 | 0.7495 | Gate clamped to $z=\mathbf{1}$. Evaluated on identical 400 synthetic test pairs. |

---

### 11.2 Answers to Critical Audit Questions

#### 1. What improved after the updates?
- **YOLOv8 Detection:** Fixed RGB vs BGR channel inversion in NumPy pipelines, eliminating false leukocyte detections and restoring true RBC counts.
- **Image Specificity:** Remediated baseline specificity from 5.0% to 63.16% using class weighting and Youden threshold calibration ($\tau^* = 0.5007$).
- **Field-of-View Contamination:** Grouped 837 smear images into 811 atomic clusters, eliminating all 14 cross-split duplicate pairs.
- **CBC Leakage Awareness:** Documented circular target leakage and proved red cell indices (`MCV, MCH, MCHC`) achieve 83.56% accuracy without Hemoglobin.
- **Missing Modality Robustness:** Implemented explicit binary masks in the GMU, enabling crash-safe inference when modalities are missing.

#### 2. What did not improve?
- **Patient-Level Linkage:** Public smear images and CBC records remain completely unlinked. No true patient-paired multimodal dataset exists in this codebase.
- **WBC Subtype Classification:** Model weights `wbc_classifier_sklearn.pth` remain missing from disk; the module relies on WHO reference baselines.

#### 3. Which previous results were affected by leakage, duplicates, or evaluation methodology?
- **Baseline Image Specificity (5.0%) & ROC-AUC (0.7477):** Distorted by severe 5.7:1 class imbalance and unpurged duplicate slides.
- **Standard CBC Accuracy (100.0%):** Caused by circular target leakage where Hemoglobin trivially reproduced its own threshold definition.
- **Old Image Split Overlap:** 14 near-duplicate candidate pairs crossed train/val/test splits in the unclustered baseline.

#### 4. Which newer results are more defensible?
- **Cluster-Remediated Image Model (0.8167 ROC-AUC):** Evaluated across 811 atomic clusters with zero cross-split leakage.
- **Leakage-Free CBC Screening (83.56% Accuracy / 0.8974 ROC-AUC):** Evaluated strictly without Hemoglobin on 809 held-out patients.
- **YOLOv8 Cell Detection (89.14% mAP50):** Evaluated on real held-out test split of 56 images from BCCD.

#### 5. Which results remain only exploratory?
- **Stacking Ensemble 99.44% Accuracy:** Derived from synthetically paired combinations of unlinked images and CBC records.
- **Experimental GMU 97.50% Accuracy:** Derived from synthetically paired combinations; demonstrates architectural gating dynamics, not clinical patient efficacy.

#### 6. What conclusions cannot currently be supported?
- That HemaVision AI is clinically validated or ready for diagnostic deployment.
- That the multimodal fusion models demonstrate diagnostic superiority on human patients over single-modality models.
- That GMU gate activations represent calibrated clinical reliability percentages.

---

## 12. Backend Architecture & API Specifications (FastAPI)

### 12.1 REST Endpoints Inventory

| Method | Path | Request Content-Type | Response Schema | Description |
|---|---|---|---|---|
| `POST` | `/api/analyze` | `multipart/form-data` | `AnalysisResponse` | Primary inference pipeline for smear images, CBC inputs, or both. |
| `GET` | `/api/history` | `None` | `List[AnalysisSummary]` | Paginated history of past analyses with limit/offset support. |
| `GET` | `/api/history/{analysis_id}` | `None` | `AnalysisDetail` | Full serialized record and parameters for single analysis. |
| `GET` | `/api/reports/{filename}` | `None` | `application/pdf` | Streams generated clinical PDF report with appropriate headers. |
| `GET` | `/api/health` | `None` | `HealthCheckResponse` | System health check and model weights verification. |

### 12.2 Actual Verified Response Payloads

#### Example 1: Dual-Modality Inference (Production Stacking Ensemble)
```json
{
  "analysis_id": "BRA-20261023201210-3B71D2",
  "timestamp": "2026-10-23T14:42:10.123456Z",
  "cell_counts": {
    "rbc": 28,
    "wbc": 2,
    "platelet": 4,
    "total": 34
  },
  "wbc_differential": {
    "neutrophils": 55.0,
    "lymphocytes": 30.0,
    "monocytes": 7.0,
    "eosinophils": 5.0,
    "basophils": 1.0,
    "strategy": "clinical_reference_fallback"
  },
  "image_anemia": {
    "prediction": "Anemic",
    "probability": 0.9124,
    "confidence": 0.9124,
    "status": "success"
  },
  "cbc_anemia": {
    "prediction": "Anemic",
    "probability": 0.9300,
    "confidence": 0.9300,
    "status": "success"
  },
  "fusion": {
    "prediction": "Anemic",
    "probability": 0.9963,
    "confidence": 0.9963,
    "strategy": "learned_stacking_ensemble",
    "model_version": "2.0.0",
    "status": "success",
    "message": "Fusion via trained Multimodal Stacking Ensemble model."
  },
  "pdf_report_url": "/api/reports/BRA-20261023201210-3B71D2.pdf"
}
```

#### Example 2: Experimental GMU Mode Selection
```json
{
  "analysis_id": "BRA-20261023203057-C18FE3",
  "timestamp": "2026-10-23T15:00:57.987654Z",
  "prediction_mode": "experimental_gmu",
  "fusion": {
    "prediction": "Anemic",
    "probability": 0.9842,
    "confidence": 0.9842,
    "strategy": "gmu_experimental",
    "model_version": "1.0.0-experimental-trained",
    "gate_scalar": 0.4499,
    "status": "success",
    "disclaimer": "RESEARCH PROTOTYPE ONLY — Trained on synthetic image-CBC combinations. Not clinically validated."
  }
}
```

---

## 13. Frontend Architectures & Clinical Reporting (Streamlit, React & FPDF2)

### 13.1 Streamlit Interface (`frontend/app.py`)
- **Workflow:** Dual-column layout for uploading blood smear images and entering CBC parameters.
- **Controls:** Sidebar radio selector switches between Production Stacking Ensemble and Experimental GMU Fusion.
- **Visuals:** Renders YOLOv8 bounding-box annotated images, cell counts, WBC differential distribution bars, and side-by-side probability gauges.
- **PDF Download:** Directly links to generated PDF report.

### 13.2 React 19 / TanStack Single Page Application (`ui/`)
- **Workflow:** Drag-and-drop upload, real-time client-side parameter validation, and asynchronous analysis execution.
- **Visuals:** Slate/navy hero container (`#0f172a` to `#1e1b4b`) displaying final verdict with bold status badges (`text-5xl font-black`), cell enumeration containers, and persistent red-bordered medical disclaimer banners.

### 13.3 FPDF2 Clinical PDF Generator (`src/report_generator.py`)
- **Structure:** Institutional header, patient demographic and CBC summary table with reference range flags, high-resolution annotated smear thumbnail, cell count summary, Matplotlib-generated WBC differential pie chart, multimodal assessment card with active strategy badge, and persistent red-bordered clinical disclaimer banner.

---

## 14. Testing, Quality Assurance & Verification History

### 14.1 Automated Test Suite Verification

The complete automated test suite was executed using Python's `unittest` framework:
```text
PS C:\Users\HP\Desktop\Blood_report_analyzer-main> .\.venv\Scripts\python -m unittest discover -s tests -p "test_*.py"
Ran 42 tests in 16.328s — OK
```

| Test File | Test Count | Scope & Covered Functions | Result |
|---|:---:|---|:---:|
| [`tests/test_cell_detection.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_cell_detection.py) | 3 | YOLO loading, RGB vs BGR channel checks, bounding box formats, tiled inference. | **PASS** |
| [`tests/test_anemia_image_model.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_anemia_image_model.py) | 2 | MobileNetV3 loading, $224\times 224$ transforms, softmax normalization, error handling. | **PASS** |
| [`tests/test_anemia_model.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_anemia_model.py) | 8 | CBC Random Forest scaling, boundaries, deterministic inference. | **PASS** |
| [`tests/test_anemia_fusion.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_anemia_fusion.py) | 6 | Stacking Ensemble, 12-d features, average fallback, single-modality routing. | **PASS** |
| [`tests/test_gmu_architecture.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_gmu_architecture.py) | 17 | GMU tensor shapes ($B\times 128$), missing-modality masks, loss gradients. | **PASS** |
| [`tests/test_gmu_experimental_inference.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_gmu_experimental_inference.py) | 2 | `predict_gmu_experimental()` inference, scaler loading, fallback handling. | **PASS** |
| [`tests/test_gmu_integration_e2e.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_gmu_integration_e2e.py) | 4 | End-to-end API integration, request dispatching, schema conformity. | **PASS** |
| **Total Test Suite** | **42** | **Full System QA Coverage** | **100% PASS** |

> [!NOTE]
> **Unit Tests vs. Scientific Validity:** A passing automated test suite verifies software stability, API contracts, tensor shapes, and crash-safety. **It does not establish clinical diagnostic accuracy on human patients.**

---

## 15. Current System State & Artifact Traceability

### 15.1 Physical Checkpoint & File Hash Registry

| Model Artifact File | Relative Path | Size | Architecture / Task Description | Operational State |
|---|---|:---:|---|:---:|
| YOLOv8 Blood Detector | `models/yolov8_blood/best.pt` | 5.94 MB | YOLOv8n fine-tuned on BCCD (Platelet, RBC, WBC) | **Active Production** |
| Cluster-Remediated Smear Model | `models/anemia_image_cluster_split/mobilenetv3_anemia_cluster_split.pth` | 6.20 MB | MobileNetV3-Small fine-tuned on 811 clusters | **Active Production** |
| Baseline Smear Model (Preserved) | `models/anemia_image/mobilenetv3_anemia_original_baseline.pth` | 6.20 MB | Unweighted baseline MobileNetV3-Small | Historical Baseline |
| CBC Random Forest Classifier | `models/anemia/anemia_model.joblib` | 1.15 MB | Random Forest Classifier on 5 CBC features | **Active Production** |
| CBC Feature Scaler | `models/anemia/scaler.joblib` | 1.80 KB | `StandardScaler` fitted on Kaggle/Master CBC | **Active Production** |
| Production Stacking Ensemble | `models/anemia_fusion/fusion_model.joblib` | 778 KB | HistGB + ExtraTrees + MLP -> LogReg Meta-Learner | **Active Production** |
| Experimental GMU Checkpoint | `models/anemia_fusion/gmu_fusion_experimental.pth` | 4.47 MB | PyTorch GMU ($d=128$, Epoch 9 checkpoint) | **Selectable Prototype** |
| Experimental GMU Scaler | `models/anemia_fusion/gmu_experimental_scaler.joblib` | 687 B | `StandardScaler` fitted on GMU train partition | **Selectable Prototype** |
| WBC Subtype Classifier | `models/resnet50_wbc/wbc_classifier_sklearn.pth` | 0 B | Scikit-learn WBC crop ensemble classifier | **MISSING (Fallback)** |

---

## 16. Threats to Validity, Failure Modes & Engineering Limitations

1. **Non-Linkage of Image and CBC Datasets:** The blood smear photomicrographs and CBC records originate from different patient cohorts with zero linking identifiers. All multimodal fusion models were trained on **synthetic combinations**. High synthetic test accuracies (99.44% and 97.50%) **do not establish clinical diagnostic efficacy on human patients**.
2. **Lack of Patient Identifiers in Smear Datasets:** While 811 atomic clusters prevent optical duplicate leakage, the underlying repositories lack patient IDs. Cluster independence does not guarantee patient independence.
3. **Stain & Optical Acquisition Variability:** Variations in Wright-Giemsa vs Leishman staining, illumination temperature, and lens aberrations across clinics have not been prospectively validated.
4. **WBC Classifier Fallback Active:** Because `wbc_classifier_sklearn.pth` is missing from disk, leukocyte differentials reflect WHO reference baselines rather than individual crop classifications.
5. **CPU Latency Constraints:** Slicing large microscopy images and running YOLOv8, MobileNetV3, and PDF compilation on commodity dual-core CPUs requires 3 to 6 seconds per sample.

---

## 17. Future Clinical Roadmap & Research Priorities

1. **Prospective Clinically Linked Cohort Collection:** Partner with teaching hospitals under IRB approval to collect 500+ patient samples with **simultaneous venipuncture collection** of digital blood smear images and five-part automated hematology analyzer CBC data.
2. **True Multimodal GMU Retraining:** Train the GMU on genuine patient pairs to evaluate real clinical diagnostic lift over single-modality models.
3. **Multi-Class Anemia Etiology Subtyping:** Extend the classification head beyond binary anemia to differentiate Iron Deficiency Anemia, $\beta$-Thalassemia trait, Megaloblastic Anemia, and Hemolytic Anemia.
4. **Quantized Edge Deployment:** Quantize YOLOv8 and MobileNetV3 to 8-bit integers (INT8) for edge deployment on low-cost Android smartphones coupled with 3D-printed microscope lens adapters.

---

## 18. Complete Reference & Artifact Inventory

1. **World Health Organization (WHO):** *Haemoglobin concentrations for the diagnosis of anaemia and assessment of severity.* Vitamin and Mineral Nutrition Information System. Geneva, World Health Organization, 2011 (WHO/NMH/NHD/MNM/11.1).
2. **Arevalo, J., Solorio, T., Montes-y-Gómez, M., & González, F. A. (2017):** *Gated Multimodal Units for Multimodal Fusion.* International Conference on Learning Representations (ICLR) Workshops.
3. **Jocher, G., Chaurasia, A., & Qiu, J. (2023):** *Ultralytics YOLOv8.* Version 8.1.0. GitHub repository: `https://github.com/ultralytics/ultralytics`.
4. **Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q. V., & Adam, H. (2019):** *Searching for MobileNetV3.* IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1314–1324.
5. **Shenggan (2019):** *BCCD (Blood Cell Count and Detection) Dataset.* MIT License. GitHub repository: `https://github.com/Shenggan/BCCD_Dataset`.
6. **Mendeley Data (2020):** *AneRBC: Peripheral Blood Smear Dataset for Anemia Classification.* Mendeley Data, V1, DOI: `10.17632/hms3sjzt7f.1`.
7. **Chulalongkorn University (2022):** *Chula-RBC-12: A Dataset of Red Blood Cell Microscopy Images with 12 Classes of Morphological Alterations.* Biomedical Informatics and Computer Vision Laboratory.
8. **Pedregosa, F., et al. (2011):** *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, pp. 2825–2830.
9. **Youden, W. J. (1950):** *Index for rating diagnostic tests.* Cancer, 3(1), pp. 32–35.

---

## 19. Evidence and Verification Status

This closing section categorizes all facts and claims documented in this report:

### A. Confirmed Facts (Supported by Local Code and Checkpoints)
- YOLOv8 nano is fine-tuned and saved at `models/yolov8_blood/best.pt`, achieving 89.14% mAP50 on BCCD.
- The RGB vs BGR channel inversion bug was resolved in `src/detect.py`.
- MobileNetV3-Small was retrained across 811 atomic clusters (`mobilenetv3_anemia_cluster_split.pth`), achieving 75.19% accuracy and 0.8167 ROC-AUC at threshold $\tau^* = 0.5007$.
- The 837 smear images partition into 811 clusters with zero cross-split leakage.
- Standard CBC models suffer from circular target leakage where Hemoglobin accounts for 80%–83.3% of feature importance.
- A leakage-free CBC Random Forest strictly without Hemoglobin achieves 83.56% accuracy and 0.8974 ROC-AUC on 809 held-out patients.
- Default fusion is a 12-feature Stacking Ensemble (`models/anemia_fusion/fusion_model.joblib`).
- The experimental GMU (`gmu_fusion_experimental.pth`) is implemented in PyTorch ($d=128$) with explicit missing-modality masks, achieving 97.50% test accuracy on 400 synthetic pairs.
- All 42 automated unit and integration tests pass cleanly (`Ran 42 tests in 16.328s, OK`).

### B. Historical Facts Supported by Records
- The original baseline smear model achieved 80.56% accuracy but suffered from 5.0% specificity collapse due to 5.7:1 class imbalance.
- The 27 near-duplicate candidate pairs were audited via dHash and cosine similarity, revealing 14 cross-split pairs in the unclustered baseline.
- Initial development used simple probability averaging fallback before the Stacking Ensemble and GMU were trained.

### C. Missing Artifacts & Historical Gaps
- `models/resnet50_wbc/wbc_classifier_sklearn.pth`: Missing from disk. The WBC differential module runs in transparent WHO reference fallback mode.
- Git Commit History: Only a single initial commit (`cbe5b3a`) exists in Git. All subsequent development exists as local workspace modifications.

### D. Known Scientific & Implementation Limitations
- Zero patient-level linkage exists between microscopic images and CBC records.
- 99.44% Stacking Ensemble and 97.50% GMU test accuracies reflect synthetic combinations and cannot be interpreted as clinical patient accuracy.
- Cluster-level independence does not guarantee patient-level independence.
- HemaVision AI is an uncertified academic research prototype and is not clinically validated.
