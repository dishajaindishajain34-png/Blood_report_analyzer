# HemaVision AI — Final Academic & Technical Engineering Report
## Multimodal Artificial Intelligence for Preliminary Anemia Screening Using Blood Smear Microscopy and Complete Blood Count Parameters

---

### ACADEMIC SUBMISSION METADATA
- **Project Title:** HemaVision AI — Multimodal AI-Based Anemia Screening Using Blood Smear Images and Complete Blood Count Parameters
- **Student / Candidate Name:** `[Candidate / Student Name]`
- **University Roll / Registration Number:** `[University Roll / Registration Number]`
- **Department / Degree:** `Department of Computer Science & Engineering, Bachelor of Technology / Master of Science`
- **Institution / University:** `[Institution / University Name]`
- **Project Supervisor / Faculty Advisor:** `[Supervisor / Faculty Advisor Name, Designation]`
- **Academic Term / Submission Date:** `Final Year Capstone Project, Academic Year 2025–2026`
- **Codebase Repository:** `https://github.com/dishajaindishajain34-png/Blood_report_analyzer`
- **Local Working Directory:** `C:\Users\HP\Desktop\Blood_report_analyzer-main`
- **System Version:** `Production v2.0.0 (with Selectable Experimental GMU v1.0.0-exp)`

---

> [!CAUTION]
> ### NON-NEGOTIABLE CLINICAL AND REGULATORY NOTICE
> **HemaVision AI is an academic research prototype for preliminary anemia screening.**  
> 1. **Not Clinically Validated:** The system has not undergone prospective multi-center clinical trials, inter-laboratory reproducibility testing, or validation on diverse human patient cohorts. It **must not** be used to make clinical diagnoses, guide therapeutic decisions, or replace laboratory testing.
> 2. **No Regulatory Clearance:** The software has **not** been submitted to, reviewed by, or cleared by the United States Food and Drug Administration (FDA) under 510(k), De Novo, or PMA pathways, nor has it received CE mark certification under European Union Medical Device Regulation (EU MDR 2017/745). It has **not** undergone a formal HIPAA compliance audit.
> 3. **Non-Linkage of Source Datasets:** The peripheral blood smear photomicrographs (`data/anerbc_raw/`, `chula_main.zip`) and tabular Complete Blood Count records (`master_cbc_dataset.csv`) originate from completely unrelated public data repositories with **zero patient-level linkage**. 
> 4. **Synthetic Nature of Multimodal Metrics:** All multimodal fusion models in this repository—including the Production Stacking Ensemble (99.44% test accuracy) and the Experimental Gated Multimodal Unit (97.50% test accuracy)—were trained and evaluated on **synthetically constructed combinations** of unlinked images and CBC records. These synthetic metrics reflect computational convergence and algorithmic gating behavior under synthetic constraints; **they do not represent real-world clinical diagnostic performance on human patients**.
> 5. **Human-in-the-Loop Requirement:** All computer vision cell detections, leukocyte differentials, and machine learning probabilities must be treated as preliminary screening indicators requiring verification by a board-certified hematologist or licensed pathologist.

---

## TABLE OF CONTENTS

1. [Audit Discrepancy & Corrections Master Table](#1-audit-discrepancy--corrections-master-table)
2. [Executive Summary & Abstract](#2-executive-summary--abstract)
3. [Clinical Hematology Fundamentals & Problem Definition](#3-clinical-hematology-fundamentals--problem-definition)
4. [System Architecture & Dual-Stream Pipeline](#4-system-architecture--dual-stream-pipeline)
5. [Datasets, Data Provenance & Split Accounting](#5-datasets-data-provenance--split-accounting)
6. [Component Implementation Verification Matrix](#6-component-implementation-verification-matrix)
7. [Blood Cell Detection & Cytomorphology (YOLOv8)](#7-blood-cell-detection--cytomorphology-yolov8)
8. [Leukocyte Differential & Reference Range Analysis](#8-leukocyte-differential--reference-range-analysis)
9. [Peripheral Blood Smear Classifier (MobileNetV3-Small)](#9-peripheral-blood-smear-classifier-mobilenetv3-small)
10. [Image Dataset Split Remediation & Cluster Evaluation](#10-image-dataset-split-remediation--cluster-evaluation)
11. [CBC Tabular Machine Learning & Circular Target Leakage](#11-cbc-tabular-machine-learning--circular-target-leakage)
12. [Leakage-Aware Anemia Screening Experiment (Without Hemoglobin)](#12-leakage-aware-anemia-screening-experiment-without-hemoglobin)
13. [Default Multimodal Fusion: Stacking Ensemble & Fallbacks](#13-default-multimodal-fusion-stacking-ensemble--fallbacks)
14. [Experimental Gated Multimodal Unit (GMU) Deep Architecture](#14-experimental-gated-multimodal-unit-gmu-deep-architecture)
15. [GMU Experimental Training Pipeline & Synthetic Protocol](#15-gmu-experimental-training-pipeline--synthetic-protocol)
16. [GMU Empirical Performance & Comparative Baselines](#16-gmu-empirical-performance--comparative-baselines)
17. [Missing-Modality Masking & Robustness Evaluation](#17-missing-modality-masking--robustness-evaluation)
18. [Software Engineering: FastAPI Backend & APIs](#18-software-engineering-fastapi-backend--apis)
19. [Frontend Architectures: Streamlit & React / TanStack Web UIs](#19-frontend-architectures-streamlit--react--tanstack-web-uis)
20. [Data Persistence & SQLite Relational Storage](#20-data-persistence--sqlite-relational-storage)
21. [Clinical PDF Generation Engine (FPDF2)](#21-clinical-pdf-generation-engine-fpdf2)
22. [Automated Test Suite & Quality Assurance](#22-automated-test-suite--quality-assurance)
23. [Metric Reconciliation & Provenance Master Table](#23-metric-reconciliation--provenance-master-table)
24. [Root Cause Analysis of Resolved Engineering Defects](#24-root-cause-analysis-of-resolved-engineering-defects)
25. [Threats to Validity & System Limitations](#25-threats-to-validity--system-limitations)
26. [Ethical Boundaries & Regulatory Reality](#26-ethical-boundaries--regulatory-reality)
27. [Future Clinical Research Directives](#27-future-clinical-research-directives)
28. [Verified Academic References & Citations](#28-verified-academic-references--citations)

---

## 1. Audit Discrepancy & Corrections Master Table

To ensure complete transparency and academic rigor, the table below documents historical discrepancies identified across earlier project summaries (including `PROJECT_REPORT.md` and initial drafts) alongside their verified ground-truth corrections grounded in physical codebase artifacts:

| Item / Claim | Earlier Report Phrasing or Value | Audited Ground Truth in Codebase | Corrected Scientific Position & Evidence | Source Artifact / File |
|---|---|---|---|---|
| **Default Fusion Model** | Described in earlier drafts as simple Logistic Regression or 50/50 average fallback. | `models/anemia_fusion/fusion_model.joblib` is an advanced 12-feature Stacking Ensemble (`StackingClassifier(cv=5)` with HistGB, ExtraTrees, and MLP). | The default runtime model is a **Stacking Ensemble Meta-Classifier**. Simple probability averaging is a **transparent runtime fallback** used when model weights or complete features are unavailable. | [`src/anemia_fusion.py:96-240`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/anemia_fusion.py#L96-L240), [`models/anemia_fusion/model_version.json`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/models/anemia_fusion/model_version.json) |
| **Default Fusion Accuracy (99.44%)** | Presented as validated multimodal performance across 2,695 patients and 859 smear images. | Evaluated on 360 held-out samples from a 2,400-sample **synthetically paired dataset** constructed by randomly combining unlinked image predictions and CBC records (`train_anemia_fusion.py:182-250`). | The 99.44% accuracy is derived from **synthetic combinations**, not real patient-linked data. It reflects the capability of the ensemble to combine synthetic feature vectors, but does not represent clinical patient accuracy. | [`scripts/train_anemia_fusion.py:182-250`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/scripts/train_anemia_fusion.py#L182-L250), [`models/anemia_fusion/metrics.json`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/models/anemia_fusion/metrics.json) |
| **Experimental GMU Accuracy (97.50%)** | Described as multimodal deep learning accuracy for anemia classification. | Evaluated on 400 held-out synthetic test pairs generated by `scripts/train_gmu_fusion.py` from disjoint image clusters and disjoint CBC IDs. | The 97.50% test accuracy demonstrates **mathematical convergence and gating dynamics on synthetic data**. It is explicitly an academic prototype limitation, not clinical diagnostic proof. | [`scripts/train_gmu_fusion.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/scripts/train_gmu_fusion.py), [`models/anemia_fusion/gmu_experimental_metrics.json`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/models/anemia_fusion/gmu_experimental_metrics.json) |
| **Patient-Level Independence** | Earlier drafts claimed patient-level independence in train/val/test splits. | AneRBC and Chula-RBC-12 image datasets provide **zero patient IDs**, clinical charts, or accession numbers. | The 837 images were partitioned into 811 atomic clusters to eliminate duplicate optical field-of-view leakage. **Cluster independence does not guarantee patient independence.** | [`scripts/audit_cluster_evaluation.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/scripts/audit_cluster_evaluation.py), [`reports/CLUSTER_EVALUATION_AUDIT_REPORT.md`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/reports/CLUSTER_EVALUATION_AUDIT_REPORT.md) |
| **Image Model ROC-AUC Discrepancy** | Contained conflicting values: `0.7477`, `0.8755`, `0.8148`, and `0.8167`. | `0.7477`: Unweighted baseline (108 test images). `0.8755`: Baseline on purged 128 images. `0.8500`: Weighted model (clerical mistranscription as 0.8148). `0.8167`: Remediated cluster-split model. | Reconciled across exact checkpoints: The active cluster-remediated model (`mobilenetv3_anemia_cluster_split.pth`) achieves **0.8167 ROC-AUC** on 129 test images across 811 atomic clusters. | [`models/anemia_image_cluster_split/reconciled_metrics.json`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/models/anemia_image_cluster_split/reconciled_metrics.json) |
| **WBC Differential Classifier** | Described as an active ResNet50/sklearn deep classifier in earlier drafts. | `src/classify_wbc.py` checks for `models/resnet50_wbc/wbc_classifier_sklearn.pth`, which is **missing from disk**, triggering fallback to WHO healthy blood ratios. | The WBC subtype classifier is **incomplete / fallback active**. The system reliably defaults to WHO reference ratios with an explicit status disclosure rather than crashing. | [`src/classify_wbc.py:61-82`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/classify_wbc.py#L61-L82) |
| **CBC Model 100% Accuracy** | Initially celebrated as perfect diagnostic machine learning accuracy. | Gini impurity analysis shows Hemoglobin accounts for **79.98% to 83.32%** of decision splits. Ground-truth labels were derived from WHO Hb cutoffs. | The model exhibits **circular target leakage**. Removing Hemoglobin reveals true morphological screening accuracy of **83.56%** (ROC-AUC: **0.8974**) using MCV, MCH, and MCHC. | [`docs/CBC_TARGET_LEAKAGE_ANALYSIS.md`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/docs/CBC_TARGET_LEAKAGE_ANALYSIS.md), [`scripts/evaluate_cbc_leakage.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/scripts/evaluate_cbc_leakage.py) |
| **Regulatory & Legal Status** | Claimed IMDRF Class IIa classification, FDA CDS qualification, and HIPAA compliance in previous report draft. | No regulatory dossier has been submitted; no accredited third-party HIPAA audit has occurred. | **All unsupported regulatory and legal assertions removed.** HemaVision AI is strictly an uncertified academic research prototype. | Non-clinical status enforced throughout this document. |
| **Report Authorship** | Previously credited to a fictional "Senior AI/ML Systems Engineer & Medical Technical Auditor". | Project represents academic student engineering work under faculty supervision. | Fictional persona replaced with **standard academic author, student registration, department, and faculty supervisor placeholders**. | Document metadata and title section. |

---

## 2. Executive Summary & Abstract

**HemaVision AI** is an academic research prototype developed to explore automated computer vision and multimodal machine learning for preliminary anemia screening and peripheral blood cytology. Anemia is a global health condition affecting over 1.9 billion individuals, with its heaviest burden in resource-constrained environments where expensive automated hematology analyzers are unavailable and manual optical microscopy is severely limited by pathologist shortages.

The system implements a decoupled dual-stream processing architecture:
1. **Cytomorphology & Cell Counting Stream:** Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) and a fine-tuned **YOLOv8n** object detector to locate and enumerate Red Blood Cells (RBCs), White Blood Cells (WBCs), and Platelets from microscopic smears. On a held-out test split of 56 images (746 cell instances), YOLOv8n achieves an overall $\text{mAP}_{50}$ of **89.14%** (Platelet: 82.74%, RBC: 85.61%, WBC: 99.06%). It incorporates a tiled inference engine with global Non-Maximum Suppression to preserve micro-thrombocytes in high-resolution fields.
2. **Peripheral Smear Anemia Stream:** Fine-tunes **MobileNetV3-Small** to extract morphological anemia indicators directly from blood smears. To eliminate field-of-view data leakage from duplicate slide crops, an automated clustering audit partitioned 837 images into **811 atomic clusters** with zero cross-split leakage. Trained with inverse-frequency class weighting and calibrated using Youden's J threshold ($\tau^* = 0.5007$), the model achieves **75.19% Accuracy, 77.27% Sensitivity, 63.16% Specificity, and 0.8167 ROC-AUC** on 129 held-out test images.
3. **Tabular Complete Blood Count (CBC) Stream:** Evaluates 5 core parameters (`[Gender, Hemoglobin, MCV, MCH, MCHC]`) using Random Forest. Our audit identified circular **target leakage** in public benchmarks where Hemoglobin accounts for 80% to 83.3% of feature importance to predict an Hb-derived label. A leakage-aware screening experiment strictly excluding Hemoglobin on 809 held-out patients demonstrated that erythrocyte cytometric indices (`MCV, MCH, MCHC`) achieve **83.56% Accuracy and 0.8974 ROC-AUC**, proving they act as viable surrogate screening markers for microcytic hypochromic anemias.
4. **Multimodal Fusion Engines:** 
   - **Production Engine (Default):** A 12-dimensional **Stacking Ensemble** combining HistGradientBoosting, ExtraTrees, and MLP base learners under a Logistic Regression meta-learner, backed by transparent probability averaging and single-modality fallbacks.
   - **Experimental Engine (Selectable):** A deep **Gated Multimodal Unit (GMU)** implemented in PyTorch ($d=128$, AdamW optimization, Best Validation Loss: 0.0617 at Epoch 9). Tested on 400 held-out synthetic combinations, the GMU achieves **97.50% Accuracy and 0.9978 ROC-AUC** with an average dynamic gate activation $\bar{z} = 0.4499$. Crucially, explicit binary missing-modality masks $(m_{\text{img}}, m_{\text{cbc}})$ mathematically prevent floating-point exceptions and enable crash-safe inference when only an image or only a lab slip is provided.

The platform is served via an asynchronous **FastAPI** backend, persists data in a local **SQLite** relational database, provides automated clinical PDF report generation via **FPDF2**, and supports both a **Streamlit** clinical dashboard and a **React 19 / TanStack** web application. All 42 automated tests pass with 100% verification.

---

## 3. Clinical Hematology Fundamentals & Problem Definition

### 3.1 The Diagnostic Challenge

In human clinical pathology, the diagnosis of anemia requires establishing whether the systemic concentration of functional hemoglobin is insufficient to meet physiological oxygen demands. Clinicians utilize two complementary diagnostic streams:
- **Complete Blood Count (CBC):** Conducted via automated five-part hematology analyzers using flow cytometry, laser light scatter, and electrical impedance. Analyzers quantify the absolute concentration of hemoglobin and measure erythrocyte physical volume and hemoglobin content. While rapid and quantitative, these analyzers cost between $15,000 and $80,000 USD, require expensive commercial reagents, and demand constant technical calibration.
- **Peripheral Blood Smear (PBS) Microscopy:** Whole blood is spread across a glass slide, stained with Romanowsky-type stains (Wright-Giemsa or Leishman), and examined under oil-immersion microscopy at 400x to 1000x magnification. A trained hematologist scans high-power fields (HPFs) to assess erythrocyte morphology, detect abnormal shapes, count platelets, and perform leukocyte differentials. However, manual microscopy is slow (15–30 minutes per slide), subject to cognitive fatigue, and suffers from 15–20% inter-observer error rates.

### 3.2 Erythrocyte Cytomorphology & Indices

Normal mature human erythrocytes are anucleated biconcave discs measuring approximately $7.5\ \mu\text{m}$ in diameter. In stained smears, they exhibit a lighter central zone occupying the middle one-third of the cell (central pallor). Pathological states distort these properties:
- **Microcytosis ($\text{MCV} < 80\ \text{fL}$):** Abnormally small red blood cells. Characteristic of Iron Deficiency Anemia (IDA) and $\beta$-Thalassemia trait.
- **Macrocytosis ($\text{MCV} > 100\ \text{fL}$):** Abnormally large red blood cells. Characteristic of Vitamin B12 and Folate deficiency (megaloblastic anemia).
- **Hypochromia ($\text{MCH} < 27\ \text{pg}$ or $\text{MCHC} < 32\ \text{g/dL}$):** Enlargement of the central pallor beyond one-third of the cell diameter, reflecting depleted cytoplasmic hemoglobin concentration.
- **Anisocytosis & Poikilocytosis:** High variability in cell dimensions (elevated Red Cell Distribution Width, RDW) and irregular morphology (e.g., target cells, tear-drop cells, sickle cells).

### 3.3 WHO Clinical Diagnostic Criteria

The World Health Organization (WHO) establishes venous Hemoglobin ($\text{Hb}$) concentration as the international diagnostic gold standard:

$$\text{Anemia} = \begin{cases} \text{Positive} & \text{if } \text{Gender} = \text{Female} \land \text{Hb} < 12.0\ \text{g/dL} \\ \text{Positive} & \text{if } \text{Gender} = \text{Male} \land \text{Hb} < 13.0\ \text{g/dL} \\ \text{Positive} & \text{if } \text{Pregnant / Child } (6\text{--}59\text{ mo}) \land \text{Hb} < 11.0\ \text{g/dL} \\ \text{Negative} & \text{otherwise} \end{cases}$$

---

## 4. System Architecture & Dual-Stream Pipeline

HemaVision AI operates as a decoupled, dual-stream processing system that processes visual cytomorphology and tabular laboratory indices in parallel before merging them through multimodal fusion.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       USER INTERACTION INTERFACES                                      │
│    Streamlit Clinical Dashboard (Port 8501)       │     React 19 / TanStack SPA (Port 8080)            │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │ HTTP Multipart POST /api/analyze
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               FASTAPI BACKEND SERVICE (backend/main.py)                                │
│                                                                                                        │
│   ┌───────────────────────────────────────────────┴─────────────────────────────────────────────┐      │
│   ▼                                                                                             ▼      │
│ ┌─────────────────────────────────────────────────────────┐   ┌──────────────────────────────────────┐ │
│ │               STREAM A: COMPUTER VISION                 │   │       STREAM B: TABULAR CBC ML       │ │
│ │               (Microscope Smear Image)                  │   │       (Laboratory Parameters)        │ │
│ │                                                         │   │                                      │ │
│ │ 1. Preprocessing (`src/preprocess.py`)                  │   │ 1. Parameter Extraction & Validation │ │
│ │    - Contrast Limited Adaptive Hist. Eq. (CLAHE)        │   │    - Gender, Hb, MCV, MCH, MCHC      │ │
│ │    - Bilateral / Gaussian smoothing                     │   │    - Boundary & range validation     │ │
│ │                                                         │   │                                      │ │
│ │ 2. Object Detection (`src/detect.py`)                   │   │ 2. Feature Scaling                   │ │
│ │    - YOLOv8n (`models/yolov8_blood/best.pt`)            │   │    - StandardScaler (`scaler.joblib`)│ │
│ │    - Classes: Platelet (0), RBC (1), WBC (2)            │   │                                      │ │
│ │    - Overlapping tiled inference for large slides       │   │ 3. Machine Learning Classification   │ │
│ │                                                         │   │    - Random Forest (`anemia_model`)  │ │
│ │ 3. WBC Differential (`src/classify_wbc.py`)             │   │    - Laboratory Probability P(cbc)   │ │
│ │    - Crop extraction & fallback to WHO reference ratios │   │    - Gini feature importances        │ │
│ │                                                         │   │                                      │ │
│ │ 4. Smear Classifier (`src/anemia_image_model.py`)       │   │                                      │ │
│ │    - MobileNetV3-Small (`mobilenetv3_anemia_cluster.pth`)│                                      │ │
│ │    - Morphological Probability P(img)                   │   │                                      │ │
│ └────────────────────────────┬────────────────────────────┘   └──────────────────┬───────────────────┘ │
│                              │                                                   │                     │
│                              └──────────────────────────┬────────────────────────┘                     │
│                                                         ▼                                              │
│ ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│ │                         MULTIMODAL ANEMIA FUSION ENGINE (`src/anemia_fusion.py`)                   │ │
│ │                                                                                                    │ │
│ │   MODE 1: Production Stacking Ensemble (Default)                                                   │ │
│ │     - Input: 12-dimensional multimodal feature vector                                              │ │
│ │     - Level-0: HistGradientBoosting + ExtraTrees + MLP                                             │ │
│ │     - Level-1: Calibrated LogisticRegression Meta-Learner (`fusion_model.joblib`)                  │ │
│ │     - Fallback Hierarchy: Probability Averaging / Single-Modality Passthrough / Insufficient Data  │ │
│ │                                                                                                    │ │
│ │   MODE 2: Experimental Gated Multimodal Unit (Selectable Prototype)                                │ │
│ │     - PyTorch GMU (`models/anemia_fusion/gmu_fusion_experimental.pth`)                             │ │
│ │     - 576-d Image Embedding + 128-d CBC Embedding -> Dynamic Sigmoid Gate z                        │ │
│ │     - Explicit Missing-Modality Binary Masks (m_img, m_cbc)                                        │ │
│ └───────────────────────────────────────────────────────┬────────────────────────────────────────────┘ │
│                                                         │                                              │
│                   ┌─────────────────────────────────────┴─────────────────────────────────────┐        │
│                   ▼                                                                           ▼        │
│ ┌───────────────────────────────────────────────────┐ ┌──────────────────────────────────────────────┐ │
│ │           DATABASE PERSISTENCE ENGINE             │ │          CLINICAL PDF GENERATOR              │ │
│ │           (`database/crud.py`, SQLite)            │ │        (`src/report_generator.py`)           │ │
│ │ - Relational record storage: `blood_analyzer.db`  │ │ - Automated clinical report compilation      │ │
│ │ - Stores analysis ID, timestamp, full JSON data   │ │ - Annotated smear thumbnail & count tables   │ │
│ │ - Independent image, CBC, fusion & GMU metrics    │ │ - WBC differential pie chart & disclaimers   │ │
│ └───────────────────────────────────────────────────┘ └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Datasets, Data Provenance & Split Accounting

### 5.1 Dataset Provenance & Structural Inventory

The machine learning and deep learning components of HemaVision AI were trained and evaluated on four distinct open-access biomedical repositories:

1. **Blood Cell Count and Detection (BCCD) Dataset:**
   - *Provenance:* Curated by Shenggan (MIT License, GitHub repository).
   - *Contents:* 364 raw peripheral blood smear microscopic fields of view annotated with bounding boxes.
   - *Annotation Counts:* 4,155 RBCs, 372 WBCs, 361 Platelets (4,888 total ground-truth instances).
   - *Splits:* Partitioned into 254 train images (expanded to 674 via 3x platelet augmentations), 54 validation images, and 56 held-out test images (746 instances).
2. **AneRBC Blood Smear Dataset:**
   - *Provenance:* Published on Mendeley Data (DOI: 10.17632/hms3sjzt7f.1).
   - *Contents:* High-power microscope photomicrographs categorized into Anemic and Normal blood smears.
3. **Chula-RBC-12 Smear Dataset:**
   - *Provenance:* Biomedical Informatics Laboratory, Chulalongkorn University.
   - *Contents:* High-resolution microscopic images of red blood cells across 12 specific morphological disease categories. 121 specialized disease images were extracted into [`data/anerbc_raw/Anemia/`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/data/anerbc_raw/Anemia/) to bring the consolidated image collection to 837 files.
4. **Master Complete Blood Count (CBC) Tabular Dataset:**
   - *Provenance:* Formed by consolidating the open-access Kaggle Anemia Dataset (1,421 records) with clinical hematology complete blood panels (`diagnosed_cbc_data_v4.csv`, 1,281 records) into [`data/anemia/processed/master_cbc_dataset.csv`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/data/anemia/processed/master_cbc_dataset.csv).
   - *Contents:* 2,695 total patient records with standardized features: `Gender, Hemoglobin, MCV, MCH, MCHC, Result, Diagnosis`.
   - *Class Distribution:* 1,429 Anemic cases (53.0%), 1,266 Normal cases (47.0%).

### 5.2 Verification of Zero Patient-Level Linkage

A critical technical audit of file headers, filenames, image metadata, and directory structures verified that **the microscopy image datasets and the tabular CBC datasets share zero patient-level linkage**:
- Image filenames consist of anonymous sequence numbers (`1.jpg`, `2.jpg` ... `837.jpg`).
- The CBC tabular dataset contains numerical patient records with anonymized row indices.
- No medical record numbers, accession IDs, laboratory chart links, or patient timestamps connect any microscopic image to any CBC row.
- **Scientific Implication:** Any multimodal model trained on both modalities simultaneously must use **synthetically generated combinations**. It is methodologically impermissible to represent synthetic benchmark accuracy as clinical diagnostic performance on real patients.

---

## 6. Component Implementation Verification Matrix

To ensure that this report represents the physical codebase exactly as it exists on disk, each feature described in this report has been audited and classified into its verified implementation state:

| System Component | Physical Codebase File(s) | Implementation State | Verification & Operational Evidence |
|---|---|:---:|---|
| **YOLOv8 Blood Cell Detector** | `src/detect.py`, `models/yolov8_blood/best.pt` | **Verified Production** | Fine-tuned YOLOv8n loads locally; detects RBC, WBC, and Platelets; passes unit tests (`test_cell_detection.py`). |
| **Tiled Inference Engine** | `src/detect.py:_detect_cells_tiled` | **Verified Production** | Slices images $>640\times 640$ into $320\times 320$ patches with 20% overlap; resolves cross-tile boxes via global NMS. |
| **WBC Subtype Classifier** | `src/classify_wbc.py:classify_wbc_crops` | **Incomplete / Fallback Active** | Code exists for sklearn classifier, but model weights `wbc_classifier_sklearn.pth` are missing from disk. Reliably defaults to WHO healthy ratios (`clinical_reference_fallback`). |
| **Smear Anemia Classifier** | `src/anemia_image_model.py`, `models/anemia_image_cluster_split/` | **Verified Production** | MobileNetV3-Small loads checkpoint `mobilenetv3_anemia_cluster_split.pth`; pre-processes $224\times 224$ images; outputs probability $P(\text{img})$. |
| **CBC Random Forest Model** | `src/anemia_model.py`, `models/anemia/anemia_model.joblib` | **Verified Production** | Loads pre-fitted `scaler.joblib` and Random Forest model; processes 5 CBC features; documents target leakage. |
| **Default Stacking Ensemble** | `src/anemia_fusion.py`, `models/anemia_fusion/fusion_model.joblib` | **Verified Production** | Loads 12-feature StackingClassifier; outputs `learned_stacking_ensemble`; transparently falls back to probability averaging if inputs/weights are missing. |
| **Experimental GMU Model** | `src/models/gmu_fusion.py`, `models/anemia_fusion/gmu_fusion_experimental.pth` | **Experimental Prototype** | PyTorch GMU ($d=128$) selectable in API/UI; loaded via `predict_gmu_experimental()`; verified missing-modality masks. |
| **FastAPI REST Service** | `backend/main.py`, `backend/routes/analyze.py` | **Verified Production** | Active on port 8000; provides `/api/analyze`, `/api/history`, `/api/reports`, and `/api/health`. |
| **Streamlit Interface** | `frontend/app.py` | **Verified Production** | Dual-column upload, live detection display, model selection radio, PDF download; active on port 8501. |
| **React / TanStack SPA** | `ui/src/routes/_app.results.$analysisId.tsx` | **Interface Verified** | Built with React 19, TypeScript, Tailwind CSS 4, and Recharts. Exists in repository under `ui/`. |
| **SQLite Persistence** | `database/models.py`, `database/crud.py` | **Verified Production** | Relational `analyses` table in `data/blood_analyzer.db`; stores full payloads, GUIDs, and file paths. |
| **Clinical PDF Generator** | `src/report_generator.py` | **Verified Production** | FPDF2 compiles multi-page clinical reports with smear thumbnail, count tables, differential chart, and safety disclaimers. |

---

## 7. Blood Cell Detection & Cytomorphology (YOLOv8)

### 7.1 Architecture & Training Hyperparameters

For real-time blood smear cytology, HemaVision AI incorporates **Ultralytics YOLOv8n (Nano)**, an anchor-free single-stage object detector:
- **Parameters & FLOPs:** 73 layers, 3,006,233 parameters, 8.1 GFLOPs.
- **Head:** Decoupled anchor-free head separating bounding box regression (CIoU loss + Distribution Focal Loss) from class score prediction (Binary Cross-Entropy).
- **Training Setup:** Fine-tuned on the BCCD dataset for 3 epochs (batch size 16, image size $320 \times 320$, AdamW optimizer with initial learning rate $\eta = 0.005$). Minority platelet instances received 3x geometric augmentations (random horizontal/vertical flips and rotations) to address class imbalance.

### 7.2 Empirical Test Performance (56 Held-Out Images, 746 Cell Instances)

| Cell Class | Ground-Truth Count | Precision ($P$) | Recall / Sensitivity ($R$) | $\text{mAP}_{50}$ | $\text{mAP}_{50-95}$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Platelet (Class 0)** | 58 | 72.47% | 81.03% | **82.74%** | 40.38% |
| **RBC (Class 1)** | 632 | 81.35% | 75.90% | **85.61%** | 55.73% |
| **WBC (Class 2)** | 56 | 96.81% | 100.00% | **99.06%** | 73.76% |
| **All Classes (Overall)** | **746** | **83.54%** | **85.65%** | **89.14%** | **56.62%** |

### 7.3 Tiled Inference for High-Power Microscopy

Microscopy images often feature high native resolutions (e.g., $2048 \times 1536$). Resizing directly to $320 \times 320$ downsamples platelets ($2\ \mu\text{m}$ diameter) into blurred sub-pixel artifacts, causing detection failure.

In [`src/detect.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/detect.py), `_detect_cells_tiled()` partitions images exceeding $640 \times 640$ into regular $320 \times 320$ patches with a 20% spatial overlap stride. Detections across all tiles are mapped back to global coordinates and filtered via global Non-Maximum Suppression ($\text{IoU}_{\text{thresh}} = 0.45$), ensuring sensitive detection of both large leukocytes and tiny thrombocytes.

---

## 8. Leukocyte Differential & Reference Range Analysis

### 8.1 Five-Part Leukocyte Differential

In peripheral blood, white blood cells are categorized into five distinct functional subtypes:
1. **Neutrophils (50–70%):** Multi-lobed nuclei (3–5 lobes) with neutral pinkish granules; first responders to acute bacterial infection.
2. **Lymphocytes (20–40%):** Large, spherical heterochromatic nuclei with thin rims of clear blue cytoplasm; mediators of adaptive cellular and humoral immunity.
3. **Monocytes (2–8%):** Large kidney-shaped or indented nuclei with abundant agranular gray-blue cytoplasm; tissue macrophage precursors.
4. **Eosinophils (1–4%):** Bilobed nuclei packed with large, refractile, bright reddish-orange granules; allergic and antiparasitic effectors.
5. **Basophils (0.5–1%):** Bilobed nuclei obscured by coarse, dark purple-black granules containing histamine and heparin.

### 8.2 Operational Status: Incomplete / Reference Fallback Active

In [`src/classify_wbc.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/classify_wbc.py), the function `classify_wbc_crops()` attempts to load an ensemble classifier from `models/resnet50_wbc/wbc_classifier_sklearn.pth`. Because this weight file is **missing from disk** in the repository, the module activates its transparent fallback protocol:
- It distributes detected WBCs according to adult WHO physiological reference proportions: Neutrophils (55.0%), Lymphocytes (30.0%), Monocytes (7.0%), Eosinophils (5.0%), and Basophils (1.0%).
- The status string is explicitly set to `"clinical_reference_fallback"`, and an explanatory notice is attached to the API response, Streamlit dashboard, and PDF report.

---

## 9. Peripheral Blood Smear Classifier (MobileNetV3-Small)

### 9.1 Model Architecture

To classify blood smear cytomorphology directly from microscopy photomicrographs, the system implements **MobileNetV3-Small** ([`src/anemia_image_model.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/anemia_image_model.py)):
- **Backbone:** Depthwise separable convolutions with Mobile Inverted Bottleneck (MBConv) blocks and hard-swish non-linearities:
  $$\text{h-swish}(x) = x \frac{\text{ReLU6}(x + 3)}{6}$$
- **Channel Attention:** Squeeze-and-Excitation (SE) modules with an attention reduction ratio of 4 dynamically recalibrate feature maps based on global context.
- **Feature Embedding:** Global average pooling produces a 576-dimensional feature vector, which passes through a dropout layer ($p=0.2$) into a linear classification head ($576 \to 2$).
- **Input Preprocessing:** $224 \times 224 \times 3$ RGB tensors normalized with ImageNet mean ($\mu = [0.485, 0.456, 0.406]$) and standard deviation ($\sigma = [0.229, 0.224, 0.225]$).

### 9.2 The Baseline Class Imbalance Defect

Initial versions of the image model were trained with standard unweighted cross-entropy loss under a default 0.5 decision threshold. Because the AneRBC dataset contained **713 Anemic images vs 124 Normal images (~5.7:1 imbalance)**, the unweighted loss led to severe classification collapse:
- The model maximized accuracy by predicting "Anemic" for nearly all samples.
- **Consequence:** While sensitivity was high (97.73%), test specificity collapsed to **5.00%** (19 False Positives out of 20 Normal test images). This defect rendered the baseline model clinically unusable without remediation.

---

## 10. Image Dataset Split Remediation & Cluster Evaluation

### 10.1 Perceptual Clustering & Elimination of Slide Leakage

Blood smear datasets frequently contain multiple photographs captured from the same physical glass slide across adjacent fields of view. When images are split randomly, fields from the same slide enter both training and test sets. Deep networks memorize the slide's unique staining intensity, illumination angle, and background tone, achieving high test accuracy via **field-of-view data leakage**.

To resolve this, an automated perceptual clustering pipeline was implemented:
- **Algorithm:** Pairwise pixel Mean Squared Error (normalized $\text{MSE} < 0.005$) and filename string sequence clustering.
- **Results:** The 837 raw images collapsed into exactly **811 atomic clusters**.
- **Candidate Pairs:** 26 confirmed duplicate/same-slide candidate pairs of identical class were identified. All 26 were grouped into indivisible atomic clusters to prevent cross-split leakage.

### 10.2 Reconciled Split Accounting

The 811 atomic clusters were split into Train, Validation, and Held-Out Test sets:

| Partition | Atomic Clusters | Total Images | Normal Images (0) | Anemic Images (1) | Cross-Split Cluster Overlap |
|---|:---:|:---:|:---:|:---:|:---:|
| **Training Split** | 567 | 583 | 86 | 497 | **0 clusters** |
| **Validation Split** | 121 | 125 | 19 | 106 | **0 clusters** |
| **Held-Out Test Split** | 123 | 129 | 19 | 110 | **0 clusters** |
| **Complete Dataset** | **811** | **837** | **124** | **713** | **0 clusters** |

$$\text{Cluster Sum Verification Check:} \quad 567 + 121 + 123 = 811 \quad (100.0\% \text{ reconciled})$$

### 10.3 Cluster-Remediated Model Evaluation & Youden Threshold Calibration

The cluster-remediated model (`models/anemia_image_cluster_split/mobilenetv3_anemia_cluster_split.pth`) was retrained using inverse-frequency loss weighting:
$$w_{\text{Normal}} = \frac{583}{2 \times 86} \approx 3.39, \quad w_{\text{Anemic}} = \frac{583}{2 \times 497} \approx 0.59$$

The operating threshold was calibrated strictly on the validation split using **Youden's J statistic**:
$$J(\tau) = \text{Sensitivity}(\tau) + \text{Specificity}(\tau) - 1 \implies \tau^* = 0.5007$$

Evaluating once on the untouched held-out test split of 129 images (123 clusters) yielded:

| Metric | Point Estimate | 95% Image Bootstrap CI | 95% Cluster-Aware Bootstrap CI | Status |
|---|:---:|:---:|:---:|:---:|
| **Accuracy** | **75.19%** | [68.20%, 82.95%] | [67.19%, 82.71%] | **Verified** |
| **Sensitivity (Recall)** | **77.27%** | [69.44%, 84.69%] | [69.37%, 85.45%] | **Verified** |
| **Specificity** | **63.16%** | [40.00%, 85.02%] | [41.17%, 84.21%] | **Verified** |
| **Precision (PPV)** | **92.39%** | [86.86%, 97.50%] | [86.21%, 97.62%] | **Verified** |
| **F1-Score** | **0.8416** | [0.7853, 0.8942] | [0.7822, 0.8959] | **Verified** |
| **ROC-AUC** | **0.8167** | [0.7220, 0.8994] | [0.7160, 0.9012] | **Verified** |

#### Test Contingency Matrix ($\tau^* = 0.5007$, 129 Test Images)
```
                          Predicted Normal (0)    Predicted Anemic (1)    Total Actual
Actual Normal (0)               12 (TN)                  7 (FP)                19
Actual Anemic (1)               25 (FN)                 85 (TP)               110
Total Predicted                 37                      92                    129
```

> [!IMPORTANT]
> **Cluster-Level vs Patient-Level Independence:** The AneRBC and Chula-RBC-12 datasets provide no patient identifiers. Grouping near-duplicates eliminates optical field-of-view leakage, but **does not guarantee patient-level independence**.

---

## 11. CBC Tabular Machine Learning & Circular Target Leakage

### 11.1 Standard Pipeline & The Target Leakage Mechanism

For Complete Blood Count analysis, the system implements a Random Forest classifier ([`src/anemia_model.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/anemia_model.py)) trained on 5 features: `[Gender, Hemoglobin, MCV, MCH, MCHC]`.

In public benchmark evaluations, this pipeline achieved near-perfect performance:
- Accuracy: **99.51% to 100.0%**
- ROC-AUC: **0.9999 to 1.0000**

#### Mathematical Explanation of Target Leakage:
1. In the Kaggle Anemia Dataset and clinical CBC repositories, ground-truth binary labels ($y \in \{0, 1\}$) were assigned by applying WHO Hemoglobin thresholds to the raw `Hemoglobin` and `Gender` columns:
   $$y = \mathbb{I}\left( (\text{Gender} = 0 \land \text{Hb} < 12.0) \lor (\text{Gender} = 1 \land \text{Hb} < 13.0) \right)$$
2. When a machine learning classifier is provided with `Hemoglobin` as an input feature to predict an Hb-derived label, the decision trees trivially identify the threshold cutoffs used to create the labels.
3. **Gini Impurity Feature Importance Proof:**
   - **Hemoglobin:** **79.98% to 83.32%**
   - Gender: **5.43%**
   - MCV: **7.00%**
   - MCH: **4.52%**
   - MCHC: **3.08%**
4. **Clinical Consequence:** The model has not learned physiological disease patterns; it has merely memorized a deterministic labeling rule. If Hemoglobin is already known from a lab analyzer, running a machine learning classifier is medically redundant.

---

## 12. Leakage-Aware Anemia Screening Experiment (Without Hemoglobin)

### 12.1 Experimental Hypothesis & Protocol

To determine whether machine learning can provide genuine clinical utility when Hemoglobin cannot be directly measured (e.g., in low-cost optical cytometers or non-invasive screening devices), an independent comparative experiment was conducted:
- **Hypothesis:** Can red blood cell cytometric indices (**MCV, MCH, MCHC**) and **Gender** predict anemia in the strict absence of Hemoglobin?
- **Dataset:** Consolidated Master CBC Dataset (`master_cbc_dataset.csv`, 2,695 patient records; 1,429 Anemic, 1,266 Normal).
- **Split:** Stratified 70% Train (1,886 records) / 30% Test (809 records held-out).
- **Comparison:**
  - *Pipeline A (Standard / Leaked):* `[Gender, Hemoglobin, MCV, MCH, MCHC]`
  - *Pipeline B (Leakage-Aware Screening):* `[Gender, MCV, MCH, MCHC]` (**strictly excluding `Hemoglobin`**)

### 12.2 Comparative Test Results (809 Held-Out Patients)

| Pipeline Configuration | Algorithm | Accuracy | Sensitivity | Specificity | F1-Score | ROC-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Pipeline A: With Hemoglobin** | Random Forest | 99.51% | 99.53% | 99.47% | 0.9953 | 0.9999 |
| *(Target Leakage Present)* | Gradient Boosting | 99.88% | 99.77% | 100.00% | 0.9988 | 1.0000 |
| | Logistic Regression | 94.81% | 94.64% | 95.00% | 0.9508 | 0.9905 |
| **Pipeline B: Without Hemoglobin** | **Random Forest** | **83.56%** | **89.28%** | **77.11%** | **0.8521** | **0.8974** |
| *(Leakage-Aware Screening)* | Gradient Boosting | 80.72% | 83.92% | 77.11% | 0.8219 | 0.8774 |
| | Logistic Regression | 64.52% | 74.83% | 52.89% | 0.6911 | 0.6646 |

### 12.3 Feature Weight Redistribution & Hematological Discussion

When `Hemoglobin` is omitted, the Random Forest model redistributes its decision splits across erythrocyte cytometric parameters:
- **MCV (Mean Corpuscular Volume):** **38.63%**
- **MCH (Mean Corpuscular Hemoglobin):** **27.77%**
- **MCHC (Mean Corpuscular Hemoglobin Conc.):** **24.08%**
- **Gender:** **9.52%**

#### Hematological Insights:
1. **Surrogate Morphological Signal:** The leakage-aware model achieves an **0.8974 ROC-AUC** because the most common causes of anemia globally—Iron Deficiency Anemia (IDA) and $\beta$-Thalassemia trait—produce microcytosis (low MCV) and hypochromia (low MCH/MCHC). These cellular morphological alterations serve as viable biological proxies for anemia.
2. **Clinical Limitations:** Normocytic normochromic anemias (e.g., aplastic anemia, early anemia of chronic disease, or acute blood loss) do not alter MCV, MCH, or MCHC, and will be missed by an Hb-free model.
3. **Medical Verdict:** While red cell indices contain non-trivial predictive signal for non-invasive screening, **they do not replace direct venous hemoglobin quantitation for definitive clinical diagnosis**.

---

## 13. Default Multimodal Fusion: Stacking Ensemble & Fallbacks

### 13.1 Production Stacking Ensemble Architecture

In production, HemaVision AI defaults to a high-capacity **Stacking Ensemble Meta-Classifier** ([`models/anemia_fusion/fusion_model.joblib`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/models/anemia_fusion/fusion_model.joblib)) that integrates 12 multimodal features:

- **12-Dimensional Feature Vector:**
  1. `image_probability`: Anemia probability from MobileNetV3-Small
  2. `image_confidence`: Distance from decision boundary: $|P_{\text{img}} - 0.5| \times 2$
  3. `image_entropy`: Information entropy: $- \sum_{c} p_c \log_2 p_c$
  4. `cbc_probability`: Anemia probability from CBC Random Forest
  5. `hemoglobin`: Venous hemoglobin (g/dL)
  6. `mcv`: Mean Corpuscular Volume (fL)
  7. `mch`: Mean Corpuscular Hemoglobin (pg)
  8. `mchc`: Mean Corpuscular Hemoglobin Concentration (g/dL)
  9. `gender`: Biological sex (0 = Female, 1 = Male)
  10. `mcv_hb_ratio`: Ratio index: $\text{MCV} / \text{Hb}$
  11. `modality_discordance`: Inter-modality discrepancy: $|P_{\text{img}} - P_{\text{cbc}}|$
  12. `modality_synergy`: Joint confidence indicator: $P_{\text{img}} \times P_{\text{cbc}}$

- **Stacking Meta-Learner Structure:**
  - *Level-0 Estimators:* `HistGradientBoostingClassifier` (150 iterations, max_depth=5), `ExtraTreesClassifier` (100 randomized trees, max_depth=6), and `MLPClassifier` (64-32 architecture, ReLU activation).
  - *Level-1 Meta-Learner:* Calibrated `LogisticRegression` trained via 5-fold cross-validation on out-of-fold probability vectors.

### 13.2 Evaluation Provenance (99.44% Accuracy on Synthetic Tabular Combinations)

On a held-out test split of 360 samples generated by `scripts/train_anemia_fusion.py` (from 2,400 total synthetic samples), the Stacking Ensemble achieved:
- **Test Accuracy:** **99.44%** | **Precision:** **99.44%** | **Sensitivity:** **99.44%** | **Specificity:** **99.44%**
- **Test ROC-AUC:** **0.9999**
- **Contingency Matrix:** $\text{TN} = 179, \text{FP} = 1, \text{FN} = 1, \text{TP} = 179$
- **Data Provenance:** As detailed in `scripts/train_anemia_fusion.py:182-250`, these samples were constructed by **randomly pairing** image prediction features from 859 smear images with CBC records from 2,695 patients. This 99.44% metric demonstrates ensemble performance on synthetic tabular combinations, **not clinical patient accuracy**.

### 13.3 Transparent Runtime Fallback Hierarchy

In [`src/anemia_fusion.py:89-240`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/anemia_fusion.py#L89-L240), `predict_fusion()` dynamically routes requests based on input completeness:
1. **Learned Stacking Ensemble (`strategy="learned_stacking_ensemble"`):** Active when both image and CBC inputs are valid and `fusion_model.joblib` is loaded.
2. **Probability Averaging Fallback (`strategy="average_fallback"`):** Active if model weights are unavailable or feature columns are incomplete:
   $$P_{\text{fused}} = \frac{P_{\text{img}} + P_{\text{cbc}}}{2}$$
3. **Single-Modality Image-Only (`strategy="image_only"`):** Active when only a smear photograph is provided ($P_{\text{fused}} = P_{\text{img}}$).
4. **Single-Modality CBC-Only (`strategy="cbc_only"`):** Active when only laboratory CBC values are provided ($P_{\text{fused}} = P_{\text{cbc}}$).
5. **Insufficient Data Guardrail (`status="insufficient_data"`):** If neither modality is provided, returns prediction `"Undetermined"` with probability $0.0$.

---

## 14. Experimental Gated Multimodal Unit (GMU) Deep Architecture

### 14.1 Mathematical Formulation of Gated Fusion

To investigate feature-level neural fusion without manual feature engineering, HemaVision AI incorporates a **Gated Multimodal Unit (GMU)** implemented in PyTorch ([`src/models/gmu_fusion.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/models/gmu_fusion.py)):

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
2. **Candidate Modality Representations:**
   $$h_{\text{img}} = \tanh(W_{\text{img}} x_{\text{img}} + b_{\text{img}}) \in (-1, 1)^{d}$$
   $$h_{\text{cbc}} = \tanh(W_{\text{cbc}} x_{\text{cbc}} + b_{\text{cbc}}) \in (-1, 1)^{d}$$
3. **Dynamic Multimodal Gating:**
   $$z = \sigma\left(W_z [h_{\text{img}}, h_{\text{cbc}}] + b_z\right) \in (0, 1)^{d}$$
   where $[h_{\text{img}}, h_{\text{cbc}}] \in \mathbb{R}^{2d}$ is the concatenated candidate vector.
4. **Gated Multimodal Combination:**
   $$h_{\text{fused}} = z \odot h_{\text{img}} + (\mathbf{1} - z) \odot h_{\text{cbc}}$$
5. **Classification Logit:**
   $$\hat{y} = \sigma\left( W_{\text{head}, 2} \left( \text{Dropout}\left(\text{Swish}\left(\text{LN}\left(W_{\text{head}, 1} h_{\text{fused}} + b_{h1}\right)\right)\right) \right) + b_{h2} \right)$$

---

## 15. GMU Experimental Training Pipeline & Synthetic Protocol

### 15.1 Synthetic Pairing Protocol & Split Isolation

To evaluate the mathematical mechanics of the GMU architecture despite the absence of patient-linked datasets, an experimental training pipeline was executed ([`scripts/train_gmu_fusion.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/scripts/train_gmu_fusion.py)):
- **Feature Extraction:** 576-d pooled visual embeddings were pre-extracted from the cluster-remediated image model (`mobilenetv3_anemia_cluster_split.pth`). CBC features were scaled via `StandardScaler` fitted strictly on training CBC rows.
- **Split Isolation:**
  - *Train Pairs (2,000):* Generated strictly from Train-split images (567 clusters) and Train-split CBCs.
  - *Validation Pairs (400):* Generated strictly from Validation-split images (121 clusters) and Validation-split CBCs.
  - *Held-Out Test Pairs (400):* Generated strictly from Held-Out Test images (123 clusters) and Held-Out Test CBCs.
  - **Zero Leakage:** No image cluster or CBC row crossed partition boundaries.

### 15.2 Discordance Protocol & Target Labeling

1. **Concordant Pairs (92%):**
   - Normal Image + Normal CBC $\implies y = 0$.
   - Anemic Image + Anemic CBC $\implies y = 1$.
2. **Discordant Pairs (8%):**
   - In clinical hematology, venous hemoglobin governs systemic diagnosis. Target labels for discordant pairs were assigned based on the WHO Hemoglobin threshold of the CBC component.
3. **Modality Masking Augmentation (20% of training samples):**
   - 10% presented with Image masked ($m_{\text{img}}=0, m_{\text{cbc}}=1$).
   - 10% presented with CBC masked ($m_{\text{img}}=1, m_{\text{cbc}}=0$).
   - 80% presented with Dual modalities ($m_{\text{img}}=1, m_{\text{cbc}}=1$).

### 15.3 Training Convergence & Checkpoint Selection

Trained via AdamW ($\eta = 0.001$, weight decay $10^{-4}$) using `BCEWithLogitsLoss()` across 10 epochs (batch size 32):
- **Epoch 1:** Train Loss 0.2630 | Val Loss 0.1029 (Acc: 97.50%)
- **Epoch 4:** Train Loss 0.1142 | Val Loss 0.0687 (Acc: 98.25%)
- **Epoch 9:** Train Loss 0.0683 | **Val Loss 0.0617 (Acc: 98.50%) — BEST CHECKPOINT SAVED**
- **Epoch 10:** Train Loss 0.0632 | Val Loss 0.0759 (Acc: 96.75%)
- The Epoch 9 checkpoint was frozen and saved to [`models/anemia_fusion/gmu_fusion_experimental.pth`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/models/anemia_fusion/gmu_fusion_experimental.pth).

---

## 16. GMU Empirical Performance & Comparative Baselines

### 16.1 Held-Out Synthetic Test Evaluation (400 Pairs)

Evaluated once on the untouched held-out synthetic test set of 400 pairs (200 Normal, 200 Anemic):

| Metric | Point Estimate | 95% Bootstrap Confidence Interval |
|---|:---:|:---:|
| **Test Accuracy** | **97.50%** | [96.00%, 98.75%] |
| **Sensitivity (Recall)** | **97.00%** | [94.50%, 99.02%] |
| **Specificity** | **98.00%** | [95.90%, 99.52%] |
| **Precision (PPV)** | **97.98%** | [95.88%, 99.52%] |
| **F1-Score** | **0.9749** | [0.9592, 0.9882] |
| **ROC-AUC** | **0.9978** | [0.9953, 0.9994] |

#### Test Contingency Matrix (Threshold = 0.50, 400 Synthetic Pairs)
```
                          Predicted Normal (0)    Predicted Anemic (1)    Total Actual
Actual Normal (0)              196 (TN)                  4 (FP)               200
Actual Anemic (1)                6 (FN)                194 (TP)               200
Total Predicted                202                     198                    400
```

### 16.2 Comparative Baseline Benchmark on Identical Synthetic Test Set

| Modality Mode / Architecture | Test Accuracy | Sensitivity | Specificity | F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Image-Only Mode** ($m_{\text{img}}=1, m_{\text{cbc}}=0$) | 69.00% | 71.00% | 67.00% | 0.6961 | 0.7495 |
| **CBC-Only Mode** ($m_{\text{img}}=0, m_{\text{cbc}}=1$) | 97.00% | 98.00% | 96.00% | 0.9703 | 0.9980 |
| **Probability Averaging Baseline** | 97.75% | 98.00% | 97.50% | 0.9776 | 0.9922 |
| **Trained GMU Dual-Modality** | **97.50%** | **97.00%** | **98.00%** | **0.9749** | **0.9978** |

### 16.3 Gating Dynamics Analysis

Across the 400 test samples, the average scalar gate activation was:
$$\bar{z} = \frac{1}{400 \times 128} \sum_{i=1}^{400} \sum_{k=1}^{128} z_{i, k} = 0.4499 \pm 0.082$$
- The network assigned an average weight of ~45% to image candidate features and ~55% to CBC candidate features.
- Under synthetic conditions where CBC parameters dictate systemic diagnosis during discordance, the GMU learned to modulate gate routing towards CBC features while retaining cytomorphological contributions.
- **Interpretation Guardrail:** Gate values $z$ represent intermediate neural routing activations across latent dimensions; they **must not** be displayed or interpreted as calibrated percentages of clinical diagnostic reliability.

---

## 17. Missing-Modality Masking & Robustness Evaluation

### 17.1 Mathematical Masking Mechanics

In real-world clinics, patients frequently present with incomplete data (a blood smear only or a CBC panel only). In [`src/models/gmu_fusion.py:112-140`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/models/gmu_fusion.py#L112-L140), the forward pass enforces binary masks $(m_{\text{img}}, m_{\text{cbc}}) \in \{0, 1\}^2$:

```python
if mask_img is not None:
    h_img = h_img * mask_img.unsqueeze(-1)
if mask_cbc is not None:
    h_cbc = h_cbc * mask_cbc.unsqueeze(-1)

if mask_img is not None and mask_cbc is not None:
    # Smear only: clamp gate z = 1.0 (100% image candidate)
    # CBC only: clamp gate z = 0.0 (100% CBC candidate)
    z = torch.where(mask_img.unsqueeze(-1) == 0, torch.zeros_like(z), z)
    z = torch.where(mask_cbc.unsqueeze(-1) == 0, torch.ones_like(z), z)
```

$$\text{Final Representation: } h_{\text{fused}} = \begin{cases} z \odot h_{\text{img}} + (\mathbf{1} - z) \odot h_{\text{cbc}} & (m_{\text{img}}=1 \land m_{\text{cbc}}=1) \\ h_{\text{img}} & (m_{\text{img}}=1 \land m_{\text{cbc}}=0 \implies z = \mathbf{1}) \\ h_{\text{cbc}} & (m_{\text{img}}=0 \land m_{\text{cbc}}=1 \implies z = \mathbf{0}) \\ \mathbf{0} & (m_{\text{img}}=0 \land m_{\text{cbc}}=0) \end{cases}$$

### 17.2 Mathematical Proof of Zero Logit Delta

Automated unit tests in [`tests/test_gmu_architecture.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_gmu_architecture.py) verify that:
1. When image is masked ($m_{\text{img}} = 0$), altering the input image tensor produces **zero change** in output logits ($\Delta \hat{y} = 0.000000$).
2. When CBC is masked ($m_{\text{cbc}} = 0$), altering input CBC values produces **zero change** in output logits ($\Delta \hat{y} = 0.000000$).
3. When both modalities are masked ($m_{\text{img}} = 0, m_{\text{cbc}} = 0$), the model returns a neutral output without generating `NaN` or `Inf` floating-point exceptions.

---

## 18. Software Engineering: FastAPI Backend & APIs

The backend is built with **FastAPI** ([`backend/main.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/backend/main.py)) and served asynchronously via Uvicorn.

### 18.1 Primary REST Endpoints

- **`POST /api/analyze`:** Main inference endpoint. Accepts multipart form data: optional smear image file, optional CBC numeric parameters, and an optional `prediction_mode` string (`"default"` vs `"experimental_gmu"`).
- **`GET /api/history`:** Queries SQLite database for historical patient analysis records with pagination (`limit`, `offset`).
- **`GET /api/history/{analysis_id}`:** Returns full serialized JSON payload for a single analysis run.
- **`GET /api/reports/{filename}`:** Streams generated clinical PDF reports from `reports/` with `application/pdf` MIME headers.
- **`GET /api/health`:** Returns operational status of all backend models, weights paths, and database connectivity.

### 18.2 Dispatching Logic in `backend/routes/analyze.py`

```python
if prediction_mode == "experimental_gmu":
    # Route to Experimental Gated Multimodal Unit
    fusion_result = predict_gmu_experimental(
        image_input=pil_image,
        cbc_features=cbc_dict
    )
else:
    # Route to Production Stacking Ensemble (Default)
    fusion_result = predict_fusion(
        image_probability=image_prob,
        cbc_probability=cbc_prob,
        cbc_features=cbc_dict
    )
```

---

## 19. Frontend Architectures: Streamlit & React / TanStack Web UIs

HemaVision AI provides two complete frontend implementations:

### 19.1 Streamlit Interface (`frontend/app.py`)
- **Interactive Controls:** Drag-and-drop microscope smear upload, numeric CBC inputs with unit validation, and a sidebar radio selector to switch between Production Stacking Ensemble and Experimental GMU Fusion.
- **Visual Rendering:** Color-coded YOLOv8 bounding boxes, WBC differential distribution bars, side-by-side probability gauges, and one-click clinical PDF download.

### 19.2 React / TanStack Web Application (`ui/`)
- **Modern Web Stack:** React 19, TypeScript, Vite 8, TanStack Router, TanStack Query, Tailwind CSS 4, and Recharts.
- **Visual Hierarchy:** Slate/navy hero card (`#0f172a` to `#1e1b4b`) displaying final verdict with bold status badges (`text-5xl font-black`), cell enumeration containers, and persistent medical disclaimer banners.

---

## 20. Data Persistence & SQLite Relational Storage

Persistence is managed via **SQLAlchemy ORM** connected to a local SQLite database at `data/blood_analyzer.db` ([`database/models.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/database/models.py), [`database/crud.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/database/crud.py)).

### Schema Overview: `analyses` Table
- `id`: Integer primary key.
- `analysis_id`: Globally unique identifier string (`BRA-YYYYMMDDHHMMSS-XXXXXX`).
- `timestamp`: ISO-8601 UTC timestamp.
- `image_path` / `annotated_image_path`: File paths to original and YOLOv8 annotated smear images.
- `rbc_count`, `wbc_count`, `platelet_count`: Absolute cell counts.
- `wbc_differential_json`: Serialized JSON storing leukocyte subtype counts and percentages.
- `image_anemia_probability`, `cbc_anemia_probability`: Raw component probabilities.
- `cbc_inputs_json`: Serialized laboratory parameters entered by the user.
- `final_prediction`, `final_probability`: Final fused prediction and probability.
- `fusion_strategy`: Active routing strategy string (`"learned_stacking_ensemble"`, `"gmu_experimental"`, `"average_fallback"`, `"image_only"`, `"cbc_only"`).
- `pdf_report_path`: File path to the compiled clinical PDF.

---

## 21. Clinical PDF Generation Engine (FPDF2)

Clinical reports are compiled programmatically using **FPDF2** ([`src/report_generator.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/report_generator.py)).
- **Sections:** Institutional header banner, patient demographic and CBC summary table with reference range flags, high-resolution annotated smear thumbnail, cell count summary, Matplotlib-generated WBC differential pie chart, multimodal anemia assessment card, and mandatory clinical disclaimer banner.
- **Asynchronous Execution:** Scheduled as a FastAPI `BackgroundTask` to ensure instant HTTP 200 API responses while PDFs compile in the background.

---

## 22. Automated Test Suite & Quality Assurance

System integrity, tensor dimensions, and routing behaviors are verified through an automated test suite across 6 dedicated test modules:

```text
PS C:\Users\HP\Desktop\Blood_report_analyzer-main> .\.venv\Scripts\python -m unittest discover -s tests -p "test_*.py"
Ran 42 tests in 16.328s — OK
```

| Test Module | Coverage Scope | Tests | Result |
|---|---|:---:|:---:|
| [`tests/test_cell_detection.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_cell_detection.py) | YOLOv8 loading, RGB/BGR channel checks, bounding box formats, tiled inference. | 3 | **PASS** |
| [`tests/test_anemia_image_model.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_anemia_image_model.py) | MobileNetV3 loading, $224\times 224$ transforms, softmax normalization, error handling. | 2 | **PASS** |
| [`tests/test_anemia_model.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_anemia_model.py) | CBC Random Forest scaling, boundaries, deterministic inference. | 8 | **PASS** |
| [`tests/test_anemia_fusion.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_anemia_fusion.py) | Stacking Ensemble, 12-d features, average fallback, single-modality routing. | 6 | **PASS** |
| [`tests/test_gmu_architecture.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_gmu_architecture.py) | GMU tensor shapes ($B\times 128$), missing-modality masks, loss gradients. | 17 | **PASS** |
| [`tests/test_gmu_integration.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_gmu_integration.py) | End-to-end `predict_gmu_experimental()`, scaler loading, API schema compliance. | 6 | **PASS** |
| **Total Automated Tests** | **Full System QA Coverage** | **42** | **100% PASS** |

---

## 23. Metric Reconciliation & Provenance Master Table

| Model & Task | Checkpoint Artifact Path | Evaluation Split / Sample Count | Documented Metric | Verified Metric | Provenance & Dataset Nature |
|---|---|---|:---:|:---:|---|
| **YOLOv8 Blood Cell Detector** | `models/yolov8_blood/best.pt` | BCCD Held-Out Test (56 images, 746 instances) | $\text{mAP}_{50} = 89.14\%$ | **89.14%** | **Real BCCD Images & Annotations.** Platelet: 82.74%, RBC: 85.61%, WBC: 99.06%. |
| **Original Baseline Smear Model** | `mobilenetv3_anemia_original_baseline.pth` | Unpurged 108 test images (AneRBC) | $\text{ROC-AUC} = 0.7477$ | **0.7477** | **Real AneRBC Images.** Collapsed specificity (5.0%) due to unweighted loss. |
| **Baseline Re-evaluation (Purged)** | `mobilenetv3_anemia_original_baseline.pth` | Purged 128 test images (AneRBC) | $\text{ROC-AUC} = 0.8755$ | **0.8752** | **Real AneRBC Images.** Cross-class conflict purged (`independent_verification.json`). |
| **Weighted Image Retraining** | `mobilenetv3_anemia.pth` | Purged 128 test images (AneRBC) | $\text{ROC-AUC} = 0.8500$ | **0.8500** | **Real AneRBC Images.** Value 0.8148 in draft table was a clerical error ($88/108$). |
| **Cluster-Remediated Image Model** | `mobilenetv3_anemia_cluster_split.pth` | Remediated 129 test images (811 clusters) | $\text{ROC-AUC} = 0.8167$ | **0.8167** | **Real AneRBC + Chula Images.** 0 cross-split leakage. Acc: 75.19%, Sens: 77.27%, Spec: 63.16%. |
| **CBC Model (Standard / Leaked)** | `models/anemia/anemia_model.joblib` | Held-Out Test (214 records, Kaggle) | $\text{Accuracy} = 100.0\%$ | **100.0%** | **Real Tabular CBCs.** Target leakage: label derived from Hb. Hb importance 83.32%. |
| **CBC Model (Master Dataset)** | `models/anemia/master_anemia_model.joblib`| Held-Out Test (809 records, Master CBC) | $\text{Accuracy} = 99.51\%$ | **99.51%** | **Real Tabular CBCs.** Target leakage present with Hb. ROC-AUC: 0.9999. |
| **CBC Model (Screening / No Hb)**| Retrained Random Forest without Hb | Held-Out Test (809 records, Master CBC) | $\text{Accuracy} = 83.56\%$ | **83.56%** | **Real Tabular CBCs.** Features: `[Gender, MCV, MCH, MCHC]`. ROC-AUC: 0.8974. |
| **Default Stacking Ensemble** | `models/anemia_fusion/fusion_model.joblib` | Held-Out Test (360 synthetic combinations) | $\text{Accuracy} = 99.44\%$ | **99.44%** | **Synthetically Paired Combinations.** Randomly paired image predictions and CBC rows. |
| **Experimental GMU Fusion** | `models/anemia_fusion/gmu_fusion_experimental.pth` | Held-Out Test (400 synthetic combinations) | $\text{Accuracy} = 97.50\%$ | **97.50%** | **Synthetically Paired Combinations.** Disjoint cluster images and CBC rows. ROC-AUC: 0.9978. |

---

## 24. Root Cause Analysis of Resolved Engineering Defects

1. **YOLOv8 RGB vs BGR Inversion:** PIL Images read in RGB order were converted to NumPy arrays and passed to YOLO, which treats NumPy inputs as BGR. Red cells appeared blue/purple and misclassified as WBCs. Resolved by passing PIL Images directly to `model.predict()` and applying `cv2.cvtColor(img, cv2.COLOR_RGB2BGR)` on all NumPy inputs.
2. **Image Classifier Specificity Collapse (5.0%):** Caused by 5.7:1 class imbalance and unweighted loss under a hardcoded 0.5 threshold. Resolved by applying inverse class frequency loss weights ($w_0 = 3.39, w_1 = 0.59$) and calibrating the operating threshold to $\tau^* = 0.5007$ using Youden's J statistic, restoring specificity to 63.16%.
3. **Cross-Split Slide Field-of-View Leakage:** Random image-level splitting placed adjacent crops from the same glass slide into both train and test splits. Resolved by clustering 837 images into 811 atomic clusters with zero cross-split overlap.
4. **Circular Target Leakage in CBC Models:** The CBC Random Forest achieved ~100% accuracy because Hemoglobin dominated feature splits (83.3%) to predict an Hb-defined label. Resolved by documenting target leakage and conducting an Hb-free screening experiment (83.56% accuracy / 0.8974 ROC-AUC).

---

## 25. Threats to Validity & System Limitations

1. **Absence of Patient-Linked Multimodal Data:** The blood smear images and CBC records originate from separate cohorts with zero linking identifiers. All multimodal fusion models were trained on **synthetic combinations**. High synthetic test accuracies (99.44% and 97.50%) **do not establish clinical diagnostic efficacy on human patients**.
2. **Lack of Patient Identifiers in Smear Datasets:** While 811 atomic clusters prevent optical duplicate leakage, the underlying repositories lack patient IDs. Cluster independence does not guarantee patient independence.
3. **Stain & Optical Acquisition Variability:** Variations in Wright-Giemsa vs Leishman staining, illumination temperature, and lens aberrations across clinics have not been prospectively validated.
4. **WBC Classifier Fallback Active:** Because `wbc_classifier_sklearn.pth` is missing from disk, leukocyte differentials reflect WHO reference baselines rather than individual crop classifications.
5. **CPU Latency Constraints:** Slicing large microscopy images and running YOLOv8, MobileNetV3, and PDF compilation on commodity dual-core CPUs requires 3 to 6 seconds per sample.

---

## 26. Ethical Boundaries & Regulatory Reality

- **Uncertified Research Prototype:** HemaVision AI has **not** been certified as a Medical Device (SaMD) under US FDA 21 CFR Part 820 / 510(k) or EU MDR 2017/745, and has **not** undergone an accredited HIPAA audit.
- **Assistive Clinical Decision Support Scope:** The system is designed strictly as an academic exploration of preliminary screening. It never dictates therapy, operates autonomously, or overrides certified pathologist review.
- **Human-in-the-Loop Mandate:** All predictions and cell counts require review and confirmation by a qualified medical practitioner.

---

## 27. Future Clinical Research Directives

1. **Prospective Clinically Linked Cohort Collection:** Partner with teaching hospitals under IRB approval to collect 500+ patient samples with **simultaneous venipuncture collection** of digital blood smear images and five-part automated hematology analyzer CBC data.
2. **True Multimodal GMU Retraining:** Train the GMU on genuine patient pairs to evaluate real clinical diagnostic lift over single-modality models.
3. **Multi-Class Anemia Etiology Subtyping:** Extend the classification head beyond binary anemia to differentiate Iron Deficiency Anemia, $\beta$-Thalassemia trait, Megaloblastic Anemia, and Hemolytic Anemia.
4. **Quantized Edge Deployment:** Quantize YOLOv8 and MobileNetV3 to 8-bit integers (INT8) for edge deployment on low-cost Android smartphones coupled with 3D-printed microscope lens adapters.

---

## 28. Verified Academic References & Citations

1. **World Health Organization (WHO):** *Haemoglobin concentrations for the diagnosis of anaemia and assessment of severity.* Vitamin and Mineral Nutrition Information System. Geneva, World Health Organization, 2011 (WHO/NMH/NHD/MNM/11.1).
2. **Arevalo, J., Solorio, T., Montes-y-Gómez, M., & González, F. A. (2017):** *Gated Multimodal Units for Multimodal Fusion.* International Conference on Learning Representations (ICLR) Workshops.
3. **Jocher, G., Chaurasia, A., & Qiu, J. (2023):** *Ultralytics YOLOv8.* Version 8.1.0. GitHub repository: `https://github.com/ultralytics/ultralytics`.
4. **Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q. V., & Adam, H. (2019):** *Searching for MobileNetV3.* IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1314–1324.
5. **Shenggan (2019):** *BCCD (Blood Cell Count and Detection) Dataset.* MIT License. GitHub repository: `https://github.com/Shenggan/BCCD_Dataset`.
6. **Mendeley Data (2020):** *AneRBC: Peripheral Blood Smear Dataset for Anemia Classification.* Mendeley Data, V1, DOI: `10.17632/hms3sjzt7f.1`.
7. **Chulalongkorn University (2022):** *Chula-RBC-12: A Dataset of Red Blood Cell Microscopy Images with 12 Classes of Morphological Alterations.* Biomedical Informatics and Computer Vision Laboratory.
8. **Pedregosa, F., et al. (2011):** *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, pp. 2825–2830.
9. **Youden, W. J. (1950):** *Index for rating diagnostic tests.* Cancer, 3(1), pp. 32–35.
