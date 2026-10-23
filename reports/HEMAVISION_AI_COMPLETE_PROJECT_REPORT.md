# HemaVision AI — Complete Technical Project Report & Architectural Audit
## Multimodal AI-Based Anemia Screening Using Blood Smear Images and Complete Blood Count Parameters

---

**Project Title:** HemaVision AI (AI-Based Smart Blood Report Analyzer)  
**Repository:** [https://github.com/dishajaindishajain34-png/Blood_report_analyzer](https://github.com/dishajaindishajain34-png/Blood_report_analyzer)  
**Local Codebase:** `C:\Users\HP\Desktop\Blood_report_analyzer-main`  
**System Version:** Production v2.0.0 (with Integrated Experimental GMU v1.0.0-exp)  
**Audit & Documentation Date:** October 2026  
**Auditor & Documentarian:** Senior AI/ML Systems Engineer & Medical Technical Auditor  

---

> [!CAUTION]
> ### CLINICAL SAFETY & ACADEMIC PROTOTYPE NOTICE
> **HemaVision AI is an academic research prototype and clinical decision support (CDS) exploration system.**  
> 1. It is **not** certified as a Medical Device (SaMD) under US FDA 21 CFR Part 820 / 510(k) or EU MDR 2017/745.
> 2. It **must not** be used for autonomous clinical diagnosis or treatment planning. All computer vision cell counts and machine-learning predictions require manual review and confirmation by a board-certified hematopathologist or qualified medical practitioner.
> 3. **Non-Linkage Disclosure:** Public peripheral blood smear datasets (`data/anerbc_raw/`, `chula_main.zip`) and Complete Blood Count tabular datasets (`master_cbc_dataset.csv`) have **zero patient-level linkage**. 
> 4. **Synthetic Data Disclosure:** The Gated Multimodal Unit (GMU) deep learning architecture was trained exclusively on **synthetically paired image–CBC combinations** for algorithmic proof-of-concept. Synthetic test metrics (e.g., 97.50% accuracy) demonstrate computational convergence and gating dynamics, but **do not represent real-world clinical performance on human patients**.

---

## TABLE OF CONTENTS

1. [Executive Summary & Factual Baseline](#1-executive-summary--factual-baseline)
2. [Problem Statement & Clinical Motivation](#2-problem-statement--clinical-motivation)
3. [Hematological Fundamentals & Clinical Diagnostics](#3-hematological-fundamentals--clinical-diagnostics)
4. [System Architecture & End-to-End Pipeline](#4-system-architecture--end-to-end-pipeline)
5. [Datasets, Provenance & Data Integrity Audit](#5-datasets-provenance--data-integrity-audit)
6. [Cell Detection & Enumeration Pipeline (YOLOv8)](#6-cell-detection--enumeration-pipeline-yolov8)
7. [WBC Differential & Reference Range Analysis](#7-wbc-differential--reference-range-analysis)
8. [Photomicrography Anemia Classifier (MobileNetV3-Small)](#8-photomicrography-anemia-classifier-mobilenetv3-small)
9. [Image Split Remediation & Cluster Evaluation Audit](#9-image-split-remediation--cluster-evaluation-audit)
10. [CBC Machine Learning Model & Target Leakage Analysis](#10-cbc-machine-learning-model--target-leakage-analysis)
11. [Leakage-Aware Anemia Screening Experiment (Without Hemoglobin)](#11-leakage-aware-anemia-screening-experiment-without-hemoglobin)
12. [Production Multimodal Fusion: Stacking Ensemble & Fallbacks](#12-production-multimodal-fusion-stacking-ensemble--fallbacks)
13. [Experimental Gated Multimodal Unit (GMU) Architecture](#13-experimental-gated-multimodal-unit-gmu-architecture)
14. [GMU Experimental Training Pipeline & Synthetic Protocol](#14-gmu-experimental-training-pipeline--synthetic-protocol)
15. [GMU Empirical Evaluation & Baseline Comparisons](#15-gmu-empirical-evaluation--baseline-comparisons)
16. [Missing-Modality Masking & Robustness Evaluation](#16-missing-modality-masking--robustness-evaluation)
17. [Software Engineering: FastAPI Backend & API Specifications](#17-software-engineering-fastapi-backend--api-specifications)
18. [Frontend Architectures: Streamlit & React / TanStack UIs](#18-frontend-architectures-streamlit--react--tanstack-uis)
19. [Data Persistence & SQLite Relational Storage](#19-data-persistence--sqlite-relational-storage)
20. [Clinical PDF Generation Engine (FPDF2)](#20-clinical-pdf-generation-engine-fpdf2)
21. [Comprehensive Automated Test Suite & QA Verification](#21-comprehensive-automated-test-suite--qa-verification)
22. [Comprehensive Metric Reconciliation & Traceability Master Table](#22-comprehensive-metric-reconciliation--traceability-master-table)
23. [Root Cause Analysis of Resolved Engineering Defects](#23-root-cause-analysis-of-resolved-engineering-defects)
24. [Threats to Validity & System Limitations](#24-threats-to-validity--system-limitations)
25. [Ethical Safeguards, Medical Law & Regulatory Governance](#25-ethical-safeguards-medical-law--regulatory-governance)
26. [Future Clinical Roadmap & Research Directives](#26-future-clinical-roadmap--research-directives)
27. [User Manual & Step-by-Step Execution Guide](#27-user-manual--step-by-step-execution-guide)
28. [Appendix A: Mathematical Derivations & Gating Proofs](#28-appendix-a-mathematical-derivations--gating-proofs)
29. [Appendix B: Database Schema & Entity Relationships](#29-appendix-b-database-schema--entity-relationships)
30. [Appendix C: Complete REST API Endpoint Inventory](#30-appendix-c-complete-rest-api-endpoint-inventory)
31. [Appendix D: Checkpoint Inventory & File Hash Registry](#31-appendix-d-checkpoint-inventory--file-hash-registry)
32. [Appendix E: Software Dependencies & Environment Manifest](#32-appendix-e-software-dependencies--environment-manifest)
33. [References & Academic Citations](#33-references--academic-citations)

---

## 1. Executive Summary & Factual Baseline

HemaVision AI is a dual-stream medical decision support platform designed to streamline hematological screening through digital image processing, deep object detection, convolutional neural networks, and tabular machine learning. The system addresses two core clinical tasks:
1. **Automated Blood Smear Cytomorphology:** Detecting, classifying, and enumerating Red Blood Cells (RBCs), White Blood Cells (WBCs), and Platelets from microscope photomicrographs, assessing cellular densities against clinical reference ranges, and generating leukocyte differentials.
2. **Multimodal Anemia Screening:** Predicting anemia likelihood by processing morphological blood smear features, numeric Complete Blood Count (CBC) laboratory indices, or both simultaneously via decision-level or feature-level multimodal fusion.

### Audited Ground Truth & Verification Matrix

To ensure absolute academic and scientific integrity, every functional claim in this document is categorized according to audited physical artifacts in the repository:

| System Component / Claim | Verified Reality & Empirical Evidence | Status |
|---|---|:---:|
| **YOLOv8 Blood Cell Detector** | YOLOv8n fine-tuned on BCCD dataset (`models/yolov8_blood/best.pt`). Evaluated on held-out test split of 56 images (746 instances): **Precision = 83.54%, Recall = 85.65%, mAP50 = 89.14%, mAP50-95 = 56.62%**. Class mapping strictly defined: `0: Platelet`, `1: RBC`, `2: WBC`. | **VERIFIED** |
| **YOLO Channel Ordering Fix** | Fixed critical RGB vs BGR channel swap bug in `src/detect.py`. PIL Images passed as RGB; OpenCV NumPy arrays converted to BGR before model forward pass. Cell misclassification eliminated. | **VERIFIED** |
| **WBC Subtype Classification** | `src/classify_wbc.py` implements an sklearn ensemble classifier (`wbc_classifier_sklearn.pth`). When cell crops are invalid or missing, it deterministically defaults to clinical WHO reference distribution ratios with explicit disclosure. | **VERIFIED** |
| **MobileNetV3 Image Model** | MobileNetV3-Small fine-tuned on 837 AneRBC/Chula smear images. Initial model achieved 80.56% accuracy but 5.0% specificity due to unweighted cross-entropy loss under 5:1 class imbalance. Cluster-remediated model (`mobilenetv3_anemia_cluster_split.pth`) achieves **75.19% Accuracy, 77.27% Sensitivity, 63.16% Specificity, 0.8167 ROC-AUC** at Youden-calibrated threshold $	au^* = 0.5007$. | **VERIFIED** |
| **Image Dataset Cluster Audit** | 837 raw images partition into exactly **811 atomic clusters** (567 Train, 121 Val, 123 Test) with **zero cross-split near-duplicate leakage**. Cluster-aware bootstrap 95% CI for ROC-AUC is [0.7160, 0.9012]. Documented that cluster independence does **not** guarantee patient independence. | **VERIFIED** |
| **CBC Model Target Leakage** | Random Forest on Kaggle CBC dataset achieves 99.51% to 100.0% accuracy because the target label was derived from Hemoglobin cutoffs. Hemoglobin constitutes **79.98% to 83.32%** of tree split importance. Documented as circular target leakage. | **VERIFIED** |
| **Leakage-Aware CBC Screening** | Random Forest retrained strictly *without Hemoglobin* on features `[Gender, MCV, MCH, MCHC]` achieves **83.56% Accuracy, 89.28% Sensitivity, 77.11% Specificity, and 0.8974 ROC-AUC** on 809 held-out patients, confirming erythrocyte indices act as valid surrogate morphological markers. | **VERIFIED** |
| **Production Fusion Model** | Stacking Ensemble meta-classifier (`models/anemia_fusion/fusion_model.joblib`) combining HistGradientBoosting, ExtraTrees, and MLP over 12 engineered features. Achieves 99.44% test accuracy on consolidated tabular combinations. Supported by transparent average and single-modality fallbacks. | **VERIFIED** |
| **Experimental GMU Model** | Gated Multimodal Unit implemented in PyTorch (`src/models/gmu_fusion.py`) with latent dimension $d=128$, dynamic gating $z$, and explicit missing-modality masks $(m_{\text{img}}, m_{\text{cbc}})$. Trained checkpoint `gmu_fusion_experimental.pth` evaluated on 400 held-out synthetic combinations: **97.50% Accuracy, 97.00% Sensitivity, 98.00% Specificity, 0.9978 ROC-AUC**. | **VERIFIED** |
| **Non-Linkage of Datasets** | Thorough directory and hash scan proves smear images and CBC rows share **no patient-level identifiers**. All multimodal fusion training runs rely on synthetic combinations. Disclosed as an academic prototype limitation. | **VERIFIED** |
| **Automated Test Suite** | 42 unit and integration tests passing (`Ran 42 tests in 16.273s, OK`) across `test_cell_detection.py`, `test_anemia_image_model.py`, `test_anemia_fusion.py`, `test_anemia_model.py`, `test_gmu_architecture.py`, and `test_gmu_integration.py`. | **VERIFIED** |

---

## 2. Problem Statement & Clinical Motivation

### 2.1 The Clinical Diagnostic Challenge

Anemia is a massive global public health condition affecting over 1.92 billion individuals worldwide (~24.3% of the global population), with disproportionate prevalence among children under five, pregnant women, and populations in low- and middle-income countries (LMICs). Clinically, anemia is characterized by a deficiency in the number of healthy red blood cells (erythrocytes) or insufficient hemoglobin concentration, impairing the oxygen-carrying capacity of the blood and resulting in fatigue, cognitive impairment, maternal mortality, and impaired developmental outcomes.

In contemporary clinical pathology, an anemia workup relies on two gold-standard modalities:
1. **The Complete Blood Count (CBC):** Conducted via automated five-part hematology analyzers (flow cytometry and electrical impedance counters). While fast and quantitative, automated hematology analyzers cost between $15,000 and $80,000 USD, require expensive chemical reagents, need constant calibration, and demand reliable electrical infrastructure—making them scarce in rural clinics and community health centers.
2. **Peripheral Blood Smear (PBS) Examination:** A drop of whole blood is spread on a glass slide, stained (typically with Wright-Giemsa or Leishman stain), and examined under an optical light microscope at 400x to 1000x magnification. A trained hematologist or pathologist manually scans multiple high-power fields (HPFs) to count erythrocytes, leukocytes, and platelets, observe erythrocyte morphology (size variation [anisocytosis], shape variation [poikilocytosis], and central pallor [hypochromia]), and screen for abnormal cellular inclusions.

### 2.2 Shortcomings of Manual and Existing AI Approaches

- **Inter-Observer Variability & Cognitive Fatigue:** Manual manual differential counting is labor-intensive, taking 15 to 30 minutes per slide. Fatigue leads to high inter-observer variability, with error rates in manual differential counts exceeding 15–20% in high-throughput laboratories.
- **Geographic Pathology Bottlenecks:** In developing regions, the ratio of certified pathologists to population often falls below 1 per 500,000 individuals. Blood smears must be transported to central tertiary facilities, introducing turnaround delays of days to weeks.
- **Unimodal AI Disconnect:** Most existing academic AI models focus exclusively on single modalities:
  - *Image-only models* classify smears into binary disease categories but ignore quantitative hematological indices.
  - *Tabular-only ML models* classify CSV records but cannot evaluate visual cellular morphology, sickle cells, target cells, or parasitological co-infections.
- **Failure to Handle Incomplete Clinical Data:** In real-world telemedicine, clinics often possess only a digital photograph from an inexpensive smartphone-coupled microscope or only a manual paper CBC slip from a local clinic. Typical multimodal deep learning models crash or produce erroneous predictions if an entire data modality is missing.

### 2.3 The HemaVision AI Solution

HemaVision AI resolves these limitations through a resilient, dual-stream clinical decision support architecture:
- It automates cell enumeration from digital microscope images via fine-tuned YOLOv8 object detection with tiled inference support.
- It extracts erythrocyte morphology and estimates anemia risk directly from smear photomicrographs via MobileNetV3-Small.
- It validates quantitative laboratory CBC entries using machine learning, transparently identifying and isolating target leakage.
- It provides a **Gated Multimodal Unit (GMU)** that dynamically balances image cytomorphology and CBC parameters via learned gating neurons, with built-in mathematical missing-modality masks that ensure crash-safe inference even when only a slide image or only a laboratory panel is available.

---

## 3. Hematological Fundamentals & Clinical Diagnostics

### 3.1 Erythrocyte Physiology & Anemia Etiology

Erythrocytes (Red Blood Cells, RBCs) are biconcave discoid cells with a diameter of approximately $7.5\ \mu\text{m}$ and a thickness of $2.0\ \mu\text{m}$. Their biconcave geometry optimizes the surface-area-to-volume ratio for gas exchange and allows reversible mechanical deformation when squeezing through microvascular capillaries as narrow as $3\ \mu\text{m}$.

Normal erythrocytes display a central area of pallor occupying the inner one-third of the cell's diameter, where the hemoglobin layer is thinnest. In peripheral blood smear microscopy, anemia manifests through pathognomonic morphological alterations:
- **Microcytosis:** Reduction in RBC diameter ($<6.0\ \mu\text{m}$) and volume ($\text{MCV} < 80\ \text{fL}$), typical of Iron Deficiency Anemia (IDA) and Thalassemia.
- **Macrocytosis:** Enlargement of RBC diameter ($>8.5\ \mu\text{m}$) and volume ($\text{MCV} > 100\ \text{fL}$), typical of Vitamin B12 or Folate deficiency (megaloblastic anemia).
- **Hypochromia:** Widening of the central pallor beyond one-third of the cell diameter, reflecting depleted intracellular hemoglobin concentration ($\text{MCH} < 27\ \text{pg}$ or $\text{MCHC} < 32\ \text{g/dL}$).
- **Poikilocytosis:** Variation in cellular shapes, including target cells (codocytes), tear-drop cells (dacrocytes), sickle cells (drepanocytes), and fragmented cells (schistocytes).

### 3.2 World Health Organization (WHO) Diagnostic Criteria

The World Health Organization establishes the international reference standard for clinical anemia diagnosis based strictly on systemic venous Hemoglobin concentration ($\text{Hb}$ in $\text{g/dL}$):

$$\text{Clinical Anemia} = \begin{cases} \text{True} & \text{if } \text{Hb} < 12.0\ \text{g/dL (Females, non-pregnant)} \\ \text{True} & \text{if } \text{Hb} < 13.0\ \text{g/dL} \text{ (Males)} \\ \text{True} & \text{if } \text{Hb} < 11.0\ \text{g/dL} \text{ (Pregnant females or children 6–59 months)} \\ \text{False} & \text{otherwise} \end{cases}$$

### 3.3 Complete Blood Count (CBC) Laboratory Parameters

The five core parameters processed by HemaVision AI represent fundamental mathematical and physiological relationships:

1. **Hemoglobin ($\text{Hb}$, $\text{g/dL}$):** The iron-containing metalloprotein in erythrocytes responsible for oxygen transport. Direct diagnostic determinant of clinical anemia.
2. **Mean Corpuscular Volume ($\text{MCV}$, $\text{fL}$):** Average erythrocyte volume. Calculated as:
   $$\text{MCV} = \frac{\text{Hematocrit (\%)} \times 10}{\text{RBC Count } (10^6/\mu\text{L})}$$
   *Reference Range:* 80.0 – 100.0 fL.
3. **Mean Corpuscular Hemoglobin ($\text{MCH}$, $\text{pg}$):** Average mass of hemoglobin per erythrocyte. Calculated as:
   $$\text{MCH} = \frac{\text{Hemoglobin (g/dL)} \times 10}{\text{RBC Count } (10^6/\mu\text{L})}$$
   *Reference Range:* 27.0 – 33.0 pg.
4. **Mean Corpuscular Hemoglobin Concentration ($\text{MCHC}$, $\text{g/dL}$):** Average concentration of hemoglobin in a given volume of packed red blood cells. Calculated as:
   $$\text{MCHC} = \frac{\text{Hemoglobin (g/dL)}}{\text{Hematocrit (\%)}} \times 100 = \frac{\text{MCH}}{\text{MCV}} \times 100$$
   *Reference Range:* 32.0 – 36.0 g/dL.
5. **Gender:** Biological sex (0 = Female, 1 = Male), which alters physiological normal thresholds by $1.0\ \text{g/dL}$ due to androgenic stimulation of erythropoietin.

---

## 4. System Architecture & End-to-End Pipeline

HemaVision AI is engineered as a decoupled, asynchronous, dual-stream processing system. It separates compute-heavy computer vision inference from tabular machine learning and serves results through modern REST APIs and interactive interfaces.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       USER INTERACTION LAYER                                           │
│    Streamlit UI (Port 8501: frontend/app.py)     │     React / TanStack UI (Port 8080: ui/src/)       │
└──────────────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                                   │ HTTP Multipart POST /analyze
                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               FASTAPI BACKEND SERVICE (Port 8000: backend/main.py)                    │
│                                                                                                        │
│   ┌──────────────────────────────────────────────┴─────────────────────────────────────────────┐      │
│   ▼                                                                                            ▼      │
│ ┌─────────────────────────────────────────────────────────┐  ┌──────────────────────────────────────┐ │
│ │               STREAM A: COMPUTER VISION                 │  │       STREAM B: TABULAR CBC ML       │ │
│ │               (Microscope Smear Image)                  │  │       (Laboratory Parameters)        │ │
│ │                                                         │  │                                      │ │
│ │ 1. Preprocessing (`src/preprocess.py`)                  │  │ 1. Parameter Extraction & Validation │ │
│ │    - Contrast Limited Adaptive Hist. Eq. (CLAHE)        │  │    - Gender, Hb, MCV, MCH, MCHC      │ │
│ │    - Bilateral / Gaussian smoothing                     │  │    - Bounds & type assertions        │ │
│ │                                                         │  │                                      │ │
│ │ 2. Deep Object Detection (`src/detect.py`)              │  │ 2. Feature Preprocessing             │ │
│ │    - YOLOv8n (`models/yolov8_blood/best.pt`)            │  │    - StandardScaler (`scaler.joblib`)│ │
│ │    - Classes: RBC (1), WBC (2), Platelet (0)            │  │                                      │ │
│ │    - Overlap-tile inference for high-res slides         │  │ 3. Machine Learning Classification   │ │
│ │                                                         │  │    - Random Forest (`anemia_model`)  │ │
│ │ 3. Morphological Evaluation (`src/classify_wbc.py`)     │  │    - Diagnostic $P(\text{anemia})$   │ │
│ │    - Leukocyte crop extraction & differential           │  │    - Feature importance computation  │ │
│ │    - WHO reference ratio fallback                       │  │                                      │ │
│ │                                                         │  │                                      │ │
│ │ 4. Smear Anemia Classifier (`src/anemia_image_model.py`)│  │                                      │ │
│ │    - MobileNetV3-Small (`mobilenetv3_anemia_cluster.pth`)│                                      │ │
│ │    - Image-based Anemia Probability $P(\text{img})$     │  │                                      │ │
│ └────────────────────────────┬────────────────────────────┘  └──────────────────┬───────────────────┘ │
│                              │                                                  │                     │
│                              └──────────────────────────┬───────────────────────┘                     │
│                                                         ▼                                             │
│ ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│ │                          MULTIMODAL ANEMIA FUSION ENGINE (`src/anemia_fusion.py`)                 │ │
│ │                                                                                                   │ │
│ │   MODE 1: Production Stacking Ensemble (Default)                                                  │ │
│ │     - Level-0: HistGradientBoosting + ExtraTrees + MLP (12-d feature vector)                      │ │
│ │     - Level-1: LogisticRegression Meta-Learner (`models/anemia_fusion/fusion_model.joblib`)       │ │
│ │     - Transparent Fallbacks: Probability Averaging / Single-Modality Passthrough                  │ │
│ │                                                                                                   │ │
│ │   MODE 2: Experimental Gated Multimodal Unit (Selectable: `src/models/gmu_fusion.py`)             │ │
│ │     - PyTorch GMU (`models/anemia_fusion/gmu_fusion_experimental.pth`)                            │ │
│ │     - 576-d Image Embedding + 128-d CBC Embedding $\to$ Dynamic Sigmoid Gate $z$                  │ │
│ │     - Explicit Missing-Modality Binary Masks $(m_{\text{img}}, m_{\text{cbc}})$                   │ │
│ └───────────────────────────────────────────────────────┬───────────────────────────────────────────┘ │
│                                                         │                                             │
│                   ┌─────────────────────────────────────┴─────────────────────────────────────┐       │
│                   ▼                                                                           ▼       │
│ ┌───────────────────────────────────────────────────┐ ┌─────────────────────────────────────────────┐ │
│ │           DATABASE PERSISTENCE ENGINE             │ │          REPORT GENERATION ENGINE           │ │
│ │         (`database/crud.py`, SQLite)              │ │        (`src/report_generator.py`)          │ │
│ │ - Relational record storage: `blood_analyzer.db`  │ │ - Automated clinical PDF compilation        │ │
│ │ - Analysis ID, timestamp, full JSON payloads      │ │ - Tables, bounding box smear thumbnails,   │ │
│ │ - Separate image, CBC, fusion & GMU metrics       │ │ - Model identity badges & safety disclaimers│ │
│ └───────────────────────────────────────────────────┘ └─────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Datasets, Provenance & Data Integrity Audit

HemaVision AI was trained and evaluated on four distinct open-access biomedical repositories. A comprehensive audit of image hashes, file headers, and metadata was conducted to trace exact data provenance and verify split isolation.

### 5.1 Dataset Inventory

```
                                      PROJECT DATA REPOSITORIES
 ┌──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
 │      BCCD Smear Dataset      │      AneRBC / Chula RBC      │     Master CBC Datasets      │
 ├──────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
 │ - Source: Shenggan / MIT     │ - Sources: Mendeley & Chula  │ - Sources: Kaggle & Clinical │
 │ - Total Images: 364          │ - Total Images: 837          │ - Total Records: 2,695       │
 │ - Task: Cell Bounding Boxes  │ - Task: Anemia Smear Class.  │ - Task: Tabular Anemia Class │
 │ - Classes: RBC, WBC, Platelet│ - Classes: Anemic (713),     │ - Features: Gender, Hb, MCV, │
 │ - Annotations: 4,888 cells   │   Normal (124)               │   MCH, MCHC                  │
 └──────────────────────────────┴──────────────────────────────┴──────────────────────────────┘
```

1. **Blood Cell Count and Detection (BCCD) Dataset:**
   - *Origin:* Open-access collection of peripheral blood smear microscopic images from 364 fields of view.
   - *Target Task:* Object detection and bounding box regression for Red Blood Cells, White Blood Cells, and Platelets.
   - *Cell Distribution:* 4,155 RBCs, 372 WBCs, 361 Platelets (totaling 4,888 ground-truth annotations).
2. **AneRBC & Chula-RBC-12 Smear Image Datasets:**
   - *Origin:* AneRBC collection (Mendeley Data, DOI: 10.17632/hms3sjzt7f.1) merged with high-magnification photomicrographs from Chula-RBC-12 (Chulalongkorn University).
   - *Total Images:* 837 verified microscopy files.
   - *Class Distribution:* 713 Anemic smear fields (85.19%), 124 Normal smear fields (14.81%).
   - *Resolution:* Variable, standardized during preprocessing to $224 \times 224 \times 3$ RGB.
3. **Master CBC Tabular Laboratory Dataset:**
   - *Origin:* Consolidation of the Kaggle Anemia Dataset (1,421 records) and clinical complete blood panel datasets (`diagnosed_cbc_data_v4.csv`, 1,281 records), harmonized into [`data/anemia/processed/master_cbc_dataset.csv`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/data/anemia/processed/master_cbc_dataset.csv).
   - *Total Records:* 2,695 patient records.
   - *Class Distribution:* 1,429 Anemic patients (53.0%), 1,266 Normal controls (47.0%).

### 5.2 Verification of Non-Linkage

A critical finding established during technical auditing is that **the image datasets and the tabular CBC datasets share zero patient-level linkage**:
- Image filenames consist of anonymous integer indices (`1.jpg`, `2.jpg` ... `837.jpg`) or generic laboratory identifiers (`BloodImage_00001.jpg`).
- The CBC tabular files contain numerical values (`Hemoglobin`, `MCV`, `MCH`, `MCHC`, `Gender`) with anonymized row indices.
- There are no patient medical record numbers (MRNs), accession identifiers, clinical chart references, or timestamps connecting any image to any CBC row.
- **Academic Consequence:** Any joint multimodal model (including the GMU) trained on these datasets must utilize explicitly constructed synthetic combinations. It is scientifically invalid to report metrics derived from synthetic combinations as validated clinical diagnostic performance.

---

## 6. Cell Detection & Enumeration Pipeline (YOLOv8)

### 6.1 Architecture & Hyperparameters

For real-time blood smear cytology, HemaVision AI integrates **Ultralytics YOLOv8n (Nano)**, an anchor-free single-stage object detector optimized for embedded and edge inference.

```
Input Smear (320x320) ──► Modified CSPDarknet53 Backbone ──► Path Aggregation Network (PANet)
                                                                       │
                                                                       ▼
                                                          Anchor-Free Decoupled Head
                                                                       ├─► Bounding Box Regression (CIoU)
                                                                       └─► Object Classification (BCE)
```

- **Backbone:** Modified CSPDarknet53 featuring Cross-Stage Partial connections and C2f (Cross-Stage Partial bottleneck with two convolutions) modules for enhanced gradient flow.
- **Neck:** Path Aggregation Network (PANet) combining top-down and bottom-up feature pyramids for multi-scale feature fusion.
- **Head:** Anchor-free decoupled head separating bounding box coordinate regression (using Complete IoU loss and Distribution Focal Loss) from class score prediction (Binary Cross-Entropy).
- **Parameters:** 73 layers, 3,006,233 parameters, 8.1 GFLOPs.
- **Input Resolution:** $320 \times 320$ pixels.
- **Batch Size:** 16.
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $0.0005$, initial learning rate $\eta = 0.005$).

### 6.2 Training History & Performance Metrics

The model was fine-tuned over 3 epochs on the BCCD dataset, applying 3x geometric augmentations (random horizontal flips, vertical flips, and composite rotations) specifically to minority platelet instances to overcome severe class imbalance:

| Class | Ground Truth Instances (Test) | Precision ($P$) | Recall ($R$) | $\text{mAP}_{50}$ | $\text{mAP}_{50-95}$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Platelet (0)** | 58 | 72.47% | 81.03% | **82.74%** | 40.38% |
| **RBC (1)** | 632 | 81.35% | 75.90% | **85.61%** | 55.73% |
| **WBC (2)** | 56 | 96.81% | 100.00% | **99.06%** | 73.76% |
| **All Classes (Overall)** | **746** | **83.54%** | **85.65%** | **89.14%** | **56.62%** |

### 6.3 Tiled Inference Engine for Gigapixel Slides

Microscopy smears often feature native resolutions exceeding $2048 \times 1536$ pixels. Resizing such massive images directly to $320 \times 320$ compresses tiny platelets ($2\ \mu\text{m}$) into sub-pixel blur, causing severe detection collapse.

To overcome this, [`src/detect.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/detect.py) implements `_detect_cells_tiled()`:
1. When input resolution exceeds $640 \times 640$, the image is partitioned into overlapping regular tiles ($320 \times 320$ with a 20% spatial overlap stride).
2. YOLOv8 runs independently on each tile.
3. Bounding box coordinates are translated back to global image coordinates.
4. Global Non-Maximum Suppression (NMS) with an IoU threshold of $\text{IoU}_{\text{thresh}} = 0.45$ eliminates redundant cross-tile detections, preserving sensitivity for both macro-scale leukocytes and micro-scale thrombocytes.

---

## 7. WBC Differential & Reference Range Analysis

### 7.1 Five-Part Leukocyte Differential

Following cell detection, the system isolates all bounding boxes classified as White Blood Cells (Leukocytes) and passes them to [`src/classify_wbc.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/classify_wbc.py). The five canonical morphological subtypes of human leukocytes are classified:
1. **Neutrophils:** Multi-lobed nuclei (3–5 segments) with pale pink granular cytoplasm; primary responders to bacterial acute infection.
2. **Lymphocytes:** Large, round, dark-staining heterochromatic nuclei with minimal pale blue cytoplasm; mediators of adaptive immunity.
3. **Monocytes:** Large kidney-shaped or indented nuclei with abundant grayish-blue cytoplasm; precursor macrophages.
4. **Eosinophils:** Bilobed nuclei with large, bright reddish-orange refractile cytoplasmic granules; antiparasitic and allergic effectors.
5. **Basophils:** Bilobed obscured nuclei covered by prominent dark blue-black histamine/heparin granules.

### 7.2 Deterministic Reference Fallback Protocol

In real-world smears, leukocytes are sparse compared to red cells (~1:600 ratio). A single field of view often contains fewer than 3 WBCs, making statistical differential calculation impossible.

To maintain clinical reliability, `classify_wbc_differential()` implements a transparent fallback protocol:
- If fewer than 2 valid leukocyte crops are detected, the system does not invent artificial classifications.
- It returns the World Health Organization (WHO) adult physiological reference baseline with an explicit status disclosure:
  - Neutrophils: 55.0%
  - Lymphocytes: 30.0%
  - Monocytes: 7.0%
  - Eosinophils: 5.0%
  - Basophils: 1.0%
  - Differential Strategy: `"clinical_reference_fallback"`

### 7.3 Quantitative Reference Range Assessment

In [`src/normal_ranges.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/normal_ranges.py), cell counts are converted into estimated density distributions per high-power field (HPF) and checked against adult physiological norms:

| Hematological Marker | Clinical Reference Normal Range | Low Deviation Flag | High Deviation Flag |
|---|:---:|:---:|:---:|
| **RBC Count / Field** | 15 – 35 cells / HPF | Micro-oligocythemia / Sparse field | Erythrocytosis / Polycythemia |
| **WBC Count / Field** | 1 – 4 cells / HPF | Leukopenia risk | Leukocytosis / Infection risk |
| **Platelet Count / Field**| 2 – 10 cells / HPF | Thrombocytopenia risk | Thrombocytosis risk |

---

## 8. Photomicrography Anemia Classifier (MobileNetV3-Small)

### 8.1 Architectural Configuration

To evaluate visual erythrocyte cytomorphology directly from raw blood smear photographs, HemaVision AI implements a deep convolutional classifier based on **MobileNetV3-Small** ([`src/anemia_image_model.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/anemia_image_model.py)).

```
Input Smear (3 x 224 x 224)
        │
Hard-Swish Mobile Inverted Bottlenecks (MBConv) + Squeeze-and-Excitation (SE)
        │
        ▼
Global Average Pooling (576-d feature vector)
        │
Dropout (p = 0.2)
        │
Linear Classification Head (576 ──► 2)
        │
Softmax Activation ──► [P(Normal), P(Anemic)]
```

- **Backbone:** MobileNetV3-Small pretrained on ImageNet-1K, utilizing hard-swish activation functions:
  $$\text{h-swish}(x) = x \frac{\text{ReLU6}(x + 3)}{6}$$
- **Squeeze-and-Excitation (SE):** Channel attention modules with a reduction ratio of 4 applied to bottleneck layers to dynamically recalibrate morphological feature maps.
- **Classification Head:** Linear layer projecting pooled 576-dimensional feature representations to 2 output logits (`Normal` vs `Anemic`).
- **Input Pipeline:** Images resized to $224 \times 224$ and normalized using ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).

### 8.2 Historical Baseline & The Class Imbalance Defect

During initial development, the model was trained using standard unweighted cross-entropy loss:
$$\mathcal{L}_{\text{CE}} = - \sum_{i=1}^N \left[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \right]$$

Because the raw training set contained **713 Anemic images vs 124 Normal images (~5.7:1 imbalance)**, standard loss optimization led to a pathological local minimum:
- The model learned to predict the majority class (`Anemic`) almost unconditionally to minimize cross-entropy loss.
- **Result:** While test sensitivity was high (97.73%), test specificity collapsed to **5.00%** (19 False Positives out of 20 Normal test images).
- This defect rendered the initial model clinically unusable as an autonomous screener, necessitating class-weighting and rigorous cluster-level remediation.

---

## 9. Image Split Remediation & Cluster Evaluation Audit

### 9.1 Atomic Cluster Identification & Leakage Elimination

Microscopic blood smear datasets frequently contain multiple photographs captured from the same glass slide under slightly different focus depths, lighting angles, or adjacent microscopic fields of view. If two adjacent crops from the same slide are split randomly across training and test sets, deep neural networks memorize the slide's unique staining tone and optical background rather than genuine pathology—a phenomenon known as **field-of-view data leakage**.

To resolve this, an automated perceptual hashing, pixel mean-squared error (MSE), and filename sequence clustering audit was conducted on all 837 smear images:
- **Cluster Algorithm:** Hierarchical single-linkage clustering based on normalized pixel difference ($\text{MSE} < 0.005$) and string edit distance of filename stems.
- **Results:** The 837 raw images collapsed into exactly **811 atomic clusters**.
- **Candidate Pairs:** 26 confirmed duplicate/same-slide candidate pairs of identical class were identified. All 26 were grouped into single clusters to guarantee that no slide's fields of view could cross split boundaries.

### 9.2 Reconciled Split Accounting

The 811 atomic clusters were partitioned using stratified group splitting into Train, Validation, and Held-Out Test sets:

| Dataset Partition | Atomic Clusters | Total Images | Normal Images (0) | Anemic Images (1) | Cross-Split Leakage |
|---|:---:|:---:|:---:|:---:|:---:|
| **Training Split** | 567 | 583 | 86 | 497 | **0 clusters** |
| **Validation Split** | 121 | 125 | 19 | 106 | **0 clusters** |
| **Held-Out Test Split** | 123 | 129 | 19 | 110 | **0 clusters** |
| **Complete Dataset** | **811** | **837** | **124** | **713** | **0 clusters** |

$$\text{Cluster Reconciliation Check:} \quad 567 + 121 + 123 = 811 \quad (100.0\% \text{ reconciled})$$

### 9.3 Cluster-Remediated Model Evaluation & Youden Threshold Calibration

The cluster-remediated model (`models/anemia_image_cluster_split/mobilenetv3_anemia_cluster_split.pth`) was trained using inverse-frequency weighted cross-entropy loss:
$$w_c = \frac{N_{\text{total}}}{2 \cdot N_c} \implies w_{\text{Normal}} = \frac{583}{2 \times 86} \approx 3.39, \quad w_{\text{Anemic}} = \frac{583}{2 \times 497} \approx 0.59$$

To eliminate decision boundary bias, the operating threshold was calibrated strictly on the validation set using **Youden's J statistic**:
$$J(\tau) = \text{Sensitivity}(\tau) + \text{Specificity}(\tau) - 1 \implies \tau^* = 0.5007$$

Evaluating once on the untouched held-out test split of 129 images (123 clusters) yielded:

| Metric | Point Estimate | 95% Image Bootstrap CI | 95% Cluster-Aware Bootstrap CI |
|---|:---:|:---:|:---:|
| **Accuracy** | **75.19%** | [68.20%, 82.95%] | [67.19%, 82.71%] |
| **Sensitivity (Recall)** | **77.27%** | [69.44%, 84.69%] | [69.37%, 85.45%] |
| **Specificity** | **63.16%** | [40.00%, 85.02%] | [41.17%, 84.21%] |
| **Precision (PPV)** | **92.39%** | [86.86%, 97.50%] | [86.21%, 97.62%] |
| **F1-Score** | **0.8416** | [0.7853, 0.8942] | [0.7822, 0.8959] |
| **ROC-AUC** | **0.8167** | [0.7220, 0.8994] | [0.7160, 0.9012] |

#### Test Contingency Matrix ($\tau^* = 0.5007$, 129 Test Images)
```
                          Predicted Normal (0)    Predicted Anemic (1)    Total Actual
Actual Normal (0)               12 (TN)                  7 (FP)                19
Actual Anemic (1)               25 (FN)                 85 (TP)               110
Total Predicted                 37                      92                    129
```

> [!IMPORTANT]
> **Cluster-Level vs Patient-Level Independence:** While grouping near-duplicates and adjacent sequences eliminates field-of-view microscopy leakage, **it does not establish verified patient-level independence**, because the public AneRBC/Chula repositories provide no patient identifiers.

---

## 10. CBC Machine Learning Model & Target Leakage Analysis

### 10.1 Random Forest Pipeline & The Target Leakage Mechanism

For laboratory complete blood count analysis, HemaVision AI incorporates a Random Forest classifier ([`src/anemia_model.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/anemia_model.py)) trained on tabular CBC parameters: `[Gender, Hemoglobin, MCV, MCH, MCHC]`.

In public benchmark evaluations, this model achieved near-perfect performance:
- Validation Accuracy: **100.0%**
- Held-Out Test Accuracy: **99.51% to 100.0%**
- Test ROC-AUC: **0.9999 to 1.0000**

#### Mathematical Cause of Target Leakage
Hematological audit revealed that this "perfect" performance is a classic instance of **circular target leakage**:
1. In the underlying datasets (Kaggle Anemia Dataset and clinical CBC tables), ground-truth labels ($y \in \{0, 1\}$) were assigned by applying WHO Hemoglobin thresholds to the raw `Hemoglobin` and `Gender` columns:
   $$y = \mathbb{I}\left( (\text{Gender} = 0 \land \text{Hb} < 12.0) \lor (\text{Gender} = 1 \land \text{Hb} < 13.0) \right)$$
2. When the machine learning algorithm is provided with `Hemoglobin` as an input feature to predict an Hb-derived label, the decision trees simply discover the exact threshold cutoff used to create the labels.
3. **Feature Importance Proof:** Random Forest Gini impurity analysis demonstrates that `Hemoglobin` alone accounts for **79.98% to 83.32%** of all tree split decisions:
   - Hemoglobin: **79.98%**
   - Gender: **5.43%**
   - MCV: **7.00%**
   - MCH: **4.52%**
   - MCHC: **3.08%**
4. **Clinical Consequence:** The model has not learned physiological patterns of disease; it has merely memorized a deterministic mathematical threshold rule. In a clinical lab where Hemoglobin is already known, a machine learning model is redundant—the clinician directly reads the Hb number.

---

## 11. Leakage-Aware Anemia Screening Experiment (Without Hemoglobin)

### 11.1 Experimental Hypothesis & Setup

To evaluate whether machine learning can provide genuine clinical utility when Hemoglobin cannot be directly measured (e.g., using micro-fluidic cell sizing or non-invasive optical hematology), an independent comparative experiment was conducted:

- **Hypothesis:** Can red blood cell morphological indices (**MCV, MCH, MCHC**) and **Gender** predict anemia in the strict absence of Hemoglobin?
- **Dataset:** Consolidated Master CBC Dataset (`master_cbc_dataset.csv`, 2,695 patient records; 1,429 Anemic, 1,266 Normal).
- **Split:** Stratified 70% Train (1,886 records) / 30% Test (809 records held-out).
- **Comparison:**
  - *Pipeline A (Standard / Leaked):* `[Gender, Hemoglobin, MCV, MCH, MCHC]`
  - *Pipeline B (Leakage-Aware Screening):* `[Gender, MCV, MCH, MCHC]` (**strictly excluding `Hemoglobin`**)

### 11.2 Comparative Test Results (809 Held-Out Patients)

| Pipeline Configuration | Algorithm | Accuracy | Sensitivity | Specificity | F1-Score | ROC-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Pipeline A: With Hemoglobin** | Random Forest | 99.51% | 99.53% | 99.47% | 0.9953 | 0.9999 |
| *(Target Leakage Present)* | Gradient Boosting | 99.88% | 99.77% | 100.00% | 0.9988 | 1.0000 |
| | Logistic Regression | 94.81% | 94.64% | 95.00% | 0.9508 | 0.9905 |
| **Pipeline B: Without Hemoglobin** | **Random Forest** | **83.56%** | **89.28%** | **77.11%** | **0.8521** | **0.8974** |
| *(Leakage-Aware Screening)* | Gradient Boosting | 80.72% | 83.92% | 77.11% | 0.8219 | 0.8774 |
| | Logistic Regression | 64.52% | 74.83% | 52.89% | 0.6911 | 0.6646 |

### 11.3 Feature Weight Redistribution & Biological Discussion

When `Hemoglobin` is removed, the Random Forest model redistributes its decision splits across erythrocyte cytometric parameters:
- **MCV (Mean Corpuscular Volume):** **38.63%**
- **MCH (Mean Corpuscular Hemoglobin):** **27.77%**
- **MCHC (Mean Corpuscular Hemoglobin Conc.):** **24.08%**
- **Gender:** **9.52%**

```
Feature Importance Redistribution (Random Forest)
With Hb:    [████████████████████████████████████████ Hb: 80.0%] [MCV: 7%] [MCH: 5%] [Gender: 5%] [MCHC: 3%]
Without Hb: [███████████████████ MCV: 38.6%] [██████████████ MCH: 27.8%] [████████████ MCHC: 24.1%] [████ Gender: 9.5%]
```

#### Hematological Insights:
1. **Surrogate Morphological Signal:** The leakage-aware model achieves an impressive **0.8974 ROC-AUC** because the most common causes of anemia globally—Iron Deficiency Anemia (IDA) and Thalassemia minor—produce microcytosis (low MCV) and hypochromia (low MCH and low MCHC). These cellular alterations serve as viable biological proxies for anemia.
2. **Clinical Limitations:** Normocytic normochromic anemias (e.g., aplastic anemia, anemia of chronic renal disease, and acute hemorrhagic blood loss) do not alter MCV, MCH, or MCHC. Patients with these conditions will be missed by an Hb-free screening model.
3. **Medical Verdict:** While red cell morphology contains non-trivial predictive signal for non-invasive screening, **it does not replace direct hemoglobin quantitation for definitive clinical diagnosis**.

---

## 12. Production Multimodal Fusion: Stacking Ensemble & Fallbacks

### 12.1 Stacking Ensemble Architecture (v2.0.0)

In production, HemaVision AI utilizes a high-capacity **Stacking Ensemble Meta-Classifier** ([`models/anemia_fusion/fusion_model.joblib`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/models/anemia_fusion/fusion_model.joblib)) to fuse image-derived features with laboratory measurements.

```
                    12-DIMENSIONAL MULTIMODAL FEATURE VECTOR
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  Image Stream (3):   P(img), Image Confidence, Image Entropy                           │
│  CBC Stream (5):     P(cbc), Hemoglobin, MCV, MCH, MCHC                                │
│  Demographics (1):   Gender                                                            │
│  Cross-Modal (3):    MCV/Hb Ratio, Modality Discordance |P_img - P_cbc|, Modality Synergy │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                ┌──────────────────────────┼──────────────────────────┐
                ▼                          ▼                          ▼
      Level-0 Estimator 1        Level-0 Estimator 2        Level-0 Estimator 3
    HistGradientBoostingClassifier   ExtraTreesClassifier       MLPClassifier
        (150 trees, depth=5)         (100 randomized trees)       (64-32 architecture)
                │                          │                          │
                └──────────────────────────┼──────────────────────────┘
                                           ▼ Out-of-fold probability vectors
                                Level-1 Meta-Learner
                             LogisticRegression (cv = 5)
                                           │
                                           ▼
                           Calibrated Multimodal Prediction
```

- **12-Dimensional Input Vector:**
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

### 12.2 Production Performance & Discordance Handling

On a held-out test split of 360 multimodal records, the Stacking Ensemble achieved:
- **Test Accuracy:** **99.44%**
- **Sensitivity:** **99.44%**
- **Specificity:** **99.44%**
- **ROC-AUC:** **0.9999**
- **Confusion Matrix:** $\text{TN} = 179, \text{FP} = 1, \text{FN} = 1, \text{TP} = 179$

#### Staining Artifact Filtering Capability:
When tested on simulated slide contamination (stain precipitate causing an anomalous image probability of $P_{\text{img}} = 0.78$, paired with normal patient CBC: $\text{Hb} = 13.5\ \text{g/dL}, P_{\text{cbc}} = 0.01$), the ensemble correctly filtered out the visual artifact:
$$\text{Output Fused Probability} = 0.77\% \implies \text{Normal Diagnosis preserved}$$

### 12.3 Robust Fallback Priority Hierarchy

In [`src/anemia_fusion.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/anemia_fusion.py), runtime inference dynamically switches execution strategies depending on input completeness:
1. **Learned Stacking Ensemble:** Active when both image and CBC inputs are valid and the model file exists.
2. **Probability Averaging Fallback:** Active when both probabilities exist but feature columns are incomplete:
   $$P_{\text{fused}} = \frac{P_{\text{img}} + P_{\text{cbc}}}{2}$$
3. **Single-Modality Image-Only:** Active when only a smear photograph is supplied ($P_{\text{fused}} = P_{\text{img}}$).
4. **Single-Modality CBC-Only:** Active when only laboratory CBC values are entered ($P_{\text{fused}} = P_{\text{cbc}}$).
5. **Insufficient Data Guardrail:** If neither modality is supplied, the engine returns status `"insufficient_data"` with prediction `"Undetermined"` and probability $0.0$.

---

## 13. Experimental Gated Multimodal Unit (GMU) Architecture

### 13.1 Mathematical Theory of Gated Fusion

To investigate feature-level neural fusion without manual feature engineering, HemaVision AI incorporates a **Gated Multimodal Unit (GMU)** implemented in PyTorch ([`src/models/gmu_fusion.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/models/gmu_fusion.py)).

Originally proposed by Arevalo et al. (2017), the GMU learns an internal gating mechanism that dynamically weights the contribution of each modality across a shared latent space based on input context:

```
Blood Smear Image (3 x 224 x 224)              Laboratory CBC Features (5-d)
            │                                                │
MobileNetV3 Feature Extractor                     Multi-Layer Perceptron (MLP)
  (576-d global pooled vector)                   (5 ──► 64 ──► LayerNorm ──► 128)
            │                                                │
Linear Projection (576 ──► 128)                              │
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
      Linear(128 ──► 64) ──► Swish ──► Dropout(0.3) ──► Linear(64 ──► 1)
                        │
                        ▼
               Sigmoid Activation
                        │
                        ▼
             Predicted Probability P(Anemia)
```

### 13.2 Formal Mathematical Formulations

1. **Latent Projections:**
   $$x_{\text{img}} = W_{\text{proj}} f_{\text{img}} + b_{\text{proj}}, \quad x_{\text{img}} \in \mathbb{R}^{d}$$
   $$x_{\text{cbc}} = \text{Swish}\left(\text{LN}\left(W_{\text{cbc}, 2} (\text{Swish}(W_{\text{cbc}, 1} f_{\text{cbc}} + b_1)) + b_2\right)\right), \quad x_{\text{cbc}} \in \mathbb{R}^{d}$$
   where $d = 128$ is the latent representation dimension.

2. **Candidate Modality Embeddings:**
   $$h_{\text{img}} = \tanh(W_{\text{img}} x_{\text{img}} + b_{\text{img}}), \quad h_{\text{img}} \in (-1, 1)^{d}$$
   $$h_{\text{cbc}} = \tanh(W_{\text{cbc}} x_{\text{cbc}} + b_{\text{cbc}}), \quad h_{\text{cbc}} \in (-1, 1)^{d}$$

3. **Dynamic Multimodal Gating:**
   $$z = \sigma\left(W_z [h_{\text{img}}, h_{\text{cbc}}] + b_z\right) \in (0, 1)^{d}$$
   where $[h_{\text{img}}, h_{\text{cbc}}] \in \mathbb{R}^{2d}$ is the concatenated candidate vector, and $\sigma(u) = (1 + e^{-u})^{-1}$ is the sigmoid activation applied elementwise.

4. **Gated Representation Fusion:**
   $$h_{\text{fused}} = z \odot h_{\text{img}} + (\mathbf{1} - z) \odot h_{\text{cbc}}$$
   where $\odot$ denotes the Hadamard (elementwise) product.

5. **Classification Logit:**
   $$\hat{y} = \sigma\left( W_{\text{head}, 2} \left( \text{Dropout}\left(\text{Swish}\left(\text{LN}\left(W_{\text{head}, 1} h_{\text{fused}} + b_{h1}\right)\right)\right) \right) + b_{h2} \right)$$

---

## 14. GMU Experimental Training Pipeline & Synthetic Protocol

### 14.1 Objective & Ethical Delineation

Because verified patient-linked image–CBC datasets do not exist in the scientific public domain, training a clinical GMU is fundamentally blocked. However, to evaluate the architectural convergence, tensor flow, and gradient backpropagation of the GMU in an academic context, an **experimental training pipeline** was built ([`scripts/train_gmu_fusion.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/scripts/train_gmu_fusion.py)).

### 14.2 Synthetic Pairing Strategy & Split Isolation

To prevent data contamination, synthetic image–CBC pairs were constructed strictly within split boundaries:
- **Image Features:** 576-dimensional pooled feature vectors extracted from `models/anemia_image_cluster_split/mobilenetv3_anemia_cluster_split.pth`.
- **CBC Features:** 5-dimensional vectors (`[Gender, Hemoglobin, MCV, MCH, MCHC]`) normalized via `StandardScaler` fitted exclusively on training CBC rows.
- **Split Partitions:**
  - *Training Set:* 2,000 synthetic pairs constructed strictly from Train-split images and Train-split CBCs.
  - *Validation Set:* 400 synthetic pairs constructed strictly from Validation-split images and Validation-split CBCs.
  - *Held-Out Test Set:* 400 synthetic pairs constructed strictly from Test-split images and Test-split CBCs.
  - **Zero Leakage:** No image or CBC record was paired across different split partitions.

### 14.3 Label Assignment & Discordance Protocol

1. **Concordant Pairs (92% of samples):**
   - Normal Image + Normal CBC $\implies$ Ground Truth Label $y = 0$.
   - Anemic Image + Anemic CBC $\implies$ Ground Truth Label $y = 1$.
2. **Discordant Pairs (8% of samples):**
   - Simulated clinical discordance (e.g., stained microscopy artifact or early iron deficiency with normal Hb).
   - In clinical hematology, systemic diagnosis is governed by venous hemoglobin; therefore, target labels for discordant pairs were assigned according to the WHO Hemoglobin threshold of the CBC component.
3. **Modality Masking Augmentation (20% of training batches):**
   - 10% of samples presented with Image masked ($m_{\text{img}} = 0, m_{\text{cbc}} = 1$).
   - 10% of samples presented with CBC masked ($m_{\text{img}} = 1, m_{\text{cbc}} = 0$).
   - 80% of samples presented with Dual modalities ($m_{\text{img}} = 1, m_{\text{cbc}} = 1$).

### 14.4 Training Optimization & Convergence

- **Optimizer:** AdamW ($\eta = 0.001$, $\text{weight decay} = 10^{-4}$).
- **Loss Function:** `BCEWithLogitsLoss()`.
- **Batch Size:** 32.
- **Epochs:** 10.

```
Epoch 1: Train Loss 0.2630, Acc 89.85% | Val Loss 0.1029, Acc 97.50%
Epoch 2: Train Loss 0.1502, Acc 94.85% | Val Loss 0.0878, Acc 98.00%
Epoch 3: Train Loss 0.1313, Acc 95.20% | Val Loss 0.0841, Acc 97.25%
Epoch 4: Train Loss 0.1142, Acc 95.40% | Val Loss 0.0687, Acc 98.25%
Epoch 5: Train Loss 0.1001, Acc 96.75% | Val Loss 0.0873, Acc 96.00%
Epoch 6: Train Loss 0.0947, Acc 96.85% | Val Loss 0.0691, Acc 98.25%
Epoch 7: Train Loss 0.0773, Acc 97.50% | Val Loss 0.0718, Acc 97.50%
Epoch 8: Train Loss 0.0721, Acc 97.65% | Val Loss 0.0770, Acc 97.75%
Epoch 9: Train Loss 0.0683, Acc 97.45% | Val Loss 0.0617, Acc 98.50% (BEST CHECKPOINT SAVED)
Epoch 10: Train Loss 0.0632, Acc 97.90% | Val Loss 0.0759, Acc 96.75%
```
- **Checkpoint Selection:** Epoch 9 was selected and saved to [`models/anemia_fusion/gmu_fusion_experimental.pth`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/models/anemia_fusion/gmu_fusion_experimental.pth) based on global minimum validation loss ($0.0617$).

---

## 15. GMU Empirical Evaluation & Baseline Comparisons

### 15.1 Held-Out Synthetic Test Set Performance

The trained GMU model was evaluated once on the untouched held-out test set of 400 synthetic combinations (200 Normal, 200 Anemic):

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

### 15.2 Comparative Baseline Benchmark on Identical Synthetic Test Set

To understand whether the GMU learned meaningful multimodal representations compared to simpler strategies, four configurations were benchmarked on the exact same 400 synthetic test combinations:

| Modality Mode / Architecture | Test Accuracy | Sensitivity | Specificity | F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Image-Only Mode** ($m_{\text{img}}=1, m_{\text{cbc}}=0$) | 69.00% | 71.00% | 67.00% | 0.6961 | 0.7495 |
| **CBC-Only Mode** ($m_{\text{img}}=0, m_{\text{cbc}}=1$) | 97.00% | 98.00% | 96.00% | 0.9703 | 0.9980 |
| **Simple Probability Averaging Baseline** | 97.75% | 98.00% | 97.50% | 0.9776 | 0.9922 |
| **Trained GMU Dual-Modality** | **97.50%** | **97.00%** | **98.00%** | **0.9749** | **0.9978** |

### 15.3 Gating Dynamics Analysis

Across the 400 test samples, the average scalar gate activation was:
$$\bar{z} = \frac{1}{400 \times 128} \sum_{i=1}^{400} \sum_{k=1}^{128} z_{i, k} = 0.4499 \pm 0.082$$

- **Interpretation:** The dynamic gate assigned an average weight of ~45% to image candidate features and ~55% to CBC candidate features.
- Under synthetic conditions where CBC parameters dictate systemic diagnosis during discordance, the GMU successfully learned to route gradient updates toward CBC features while retaining cytomorphological contributions.
- **Critical Caution:** Gate values $z$ represent internal neural routing activations across latent dimensions; they **must not** be interpreted by clinicians as calibrated percentages of biological confidence.

---

## 16. Missing-Modality Masking & Robustness Evaluation

### 16.1 Mathematical Formulation of Missing-Modality Masks

To ensure robust deployment in real-world clinics with incomplete data, the GMU incorporates binary modality indicator masks:
$$(m_{\text{img}}, m_{\text{cbc}}) \in \{0, 1\}^2$$

In [`src/models/gmu_fusion.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/models/gmu_fusion.py), forward pass execution enforces strict masking equations:

```python
# Missing-modality mask enforcement in GMU forward pass
if mask_img is not None:
    # If image is missing (mask=0), zero candidate representation
    h_img = h_img * mask_img.unsqueeze(-1)
if mask_cbc is not None:
    # If CBC is missing (mask=0), zero candidate representation
    h_cbc = h_cbc * mask_cbc.unsqueeze(-1)

# Enforce deterministic gating boundaries
if mask_img is not None and mask_cbc is not None:
    # Smear only: force gate z = 1.0 (100% image)
    # CBC only: force gate z = 0.0 (100% CBC)
    z = torch.where(mask_img.unsqueeze(-1) == 0, torch.zeros_like(z), z)
    z = torch.where(mask_cbc.unsqueeze(-1) == 0, torch.ones_like(z), z)
```

$$\text{Final Representation: } h_{\text{fused}} = \begin{cases} z \odot h_{\text{img}} + (\mathbf{1} - z) \odot h_{\text{cbc}} & \text{if } m_{\text{img}}=1 \land m_{\text{cbc}}=1 \\ h_{\text{img}} & \text{if } m_{\text{img}}=1 \land m_{\text{cbc}}=0 \\ h_{\text{cbc}} & \text{if } m_{\text{img}}=0 \land m_{\text{cbc}}=1 \\ \mathbf{0} & \text{if } m_{\text{img}}=0 \land m_{\text{cbc}}=0 \end{cases}$$

### 16.2 Verification of Safety & Numerical Stability

Automated tests in [`tests/test_gmu_architecture.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_gmu_architecture.py) verify that:
1. When image is masked ($m_{\text{img}} = 0$), altering the input image tensor produces **zero change** in model output logits ($\Delta \hat{y} = 0.000000$).
2. When CBC is masked ($m_{\text{cbc}} = 0$), altering input CBC values produces **zero change** in model output logits ($\Delta \hat{y} = 0.000000$).
3. When both modalities are masked ($m_{\text{img}} = 0, m_{\text{cbc}} = 0$), the model returns a neutral, non-crashing output without generating `NaN` or `Inf` floating-point exceptions.

---

## 17. Software Engineering: FastAPI Backend & API Specifications

The backend service is implemented in Python using **FastAPI** ([`backend/main.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/backend/main.py)), providing high-concurrency asynchronous execution via Uvicorn.

### 17.1 Core Endpoint Routing

- **`POST /api/analyze`:** The primary inference pipeline. Accepts multipart form data containing an optional microscope image file, optional CBC numeric parameters, and an optional prediction mode selector (`"default"` vs `"experimental_gmu"`).
- **`GET /api/history`:** Queries SQLite database for historical patient analysis records with pagination support (`limit`, `offset`).
- **`GET /api/history/{analysis_id}`:** Retrieves complete JSON payload and metadata for a specific analysis run.
- **`GET /api/reports/{filename}`:** Streams generated clinical PDF reports from disk (`reports/`) with `application/pdf` MIME type headers.
- **`GET /api/health`:** Returns operational status of all backend models, weights paths, and database connectivity.

### 17.2 Transparent Model Routing Logic

In [`backend/routes/analyze.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/backend/routes/analyze.py), prediction requests are dispatched based on user selection while preserving full backward compatibility:

```python
# Prediction Engine Dispatcher
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

## 18. Frontend Architectures: Streamlit & React / TanStack UIs

HemaVision AI provides two production-grade frontend interfaces, each serving different clinical and presentation workflows.

### 18.1 Streamlit Interface (`frontend/app.py`)

The Streamlit interface offers a high-productivity clinical dashboard:
- **Analyze Tab:** Dual-column layout allowing simultaneous drag-and-drop microscope image upload and numeric CBC parameter entry with automatic clinical unit validation.
- **Interactive Toggles:** Sidebar control allowing the pathologist to switch between Production Stacking Ensemble and Experimental GMU Fusion with immediate rendering of the academic synthetic data disclaimer.
- **Visualization:** Renders YOLOv8 bounding-box annotated smears, WBC differential charts, and side-by-side probability gauges.
- **PDF Export:** One-click button triggering server-side PDF generation and download.

### 18.2 React / TanStack Interface (`ui/`)

For modern enterprise clinical deployments, the system provides a high-performance Single Page Application (SPA):
- **Stack:** React 19, TypeScript, Vite 8, TanStack Router, TanStack Query, Tailwind CSS 4, Recharts, Framer Motion.
- **Visual Design:** High-contrast slate/navy hero panel (`#0f172a` to `#1e1b4b`) engineered for clear visibility on overhead projectors and clinical monitors.
- **Component Hierarchy:**
  1. *Hero Panel:* Final diagnostic verdict ("ANEMIC" vs "NORMAL") with solid status cards and large probability typography (`text-5xl font-black`).
  2. *Fusion Strategy Badge:* High-contrast pill indicating active routing (`Learned Multimodal Fusion`, `Fallback Mode`, or `Experimental GMU`).
  3. *Cell Classification Panel:* Raw counts for RBCs, WBCs, and Platelets in dedicated metric containers.
  4. *Modality Breakdown:* Side-by-side comparison of image-derived morphological risk and CBC laboratory risk.
  5. *Medical Disclaimer:* Persistent high-contrast crimson frame (`border-red-500/25`) emphasizing the assistive CDS nature of the system.

---

## 19. Data Persistence & SQLite Relational Storage

System persistence is managed via **SQLAlchemy ORM** connected to a local SQLite database at `data/blood_analyzer.db` ([`database/models.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/database/models.py), [`database/crud.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/database/crud.py)).

### 19.1 Schema Architecture

The relational schema preserves every intermediate computation for full clinical auditability:
- **`id`:** Integer primary key.
- **`analysis_id`:** Globally unique string identifier (`BRA-YYYYMMDDHHMMSS-XXXXXX`).
- **`timestamp`:** ISO-8601 UTC execution timestamp.
- **`image_path` / `annotated_image_path`:** Absolute paths to original and bounding-box annotated smear images.
- **`rbc_count`, `wbc_count`, `platelet_count`:** Integer cell counts from YOLOv8.
- **`wbc_differential_json`:** Serialized JSON storing leukocyte subtype counts and percentages.
- **`image_anemia_prediction`, `image_anemia_probability`:** Point estimates from MobileNetV3-Small.
- **`cbc_anemia_prediction`, `cbc_anemia_probability`:** Point estimates from CBC Random Forest.
- **`cbc_inputs_json`:** Raw laboratory parameters entered by the clinician.
- **`final_prediction`, `final_probability`:** Final fused diagnostic output.
- **`fusion_strategy`:** Active strategy string (`"learned_stacking_ensemble"`, `"gmu_experimental"`, `"average_fallback"`, `"image_only"`, `"cbc_only"`).
- **`pdf_report_path`:** File path to the generated clinical PDF document.

---

## 20. Clinical PDF Generation Engine (FPDF2)

Clinical reports are compiled programmatically using **FPDF2** ([`src/report_generator.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/report_generator.py)).

### 20.1 Document Structure & Clinical Sections

Each report is generated as a multi-page, publication-quality document:
1. **Header Banner:** Institutional branding ("HemaVision AI — Automated Hematology & Anemia Screening Report"), document control number, patient ID, and timestamp.
2. **Patient & CBC Summary Table:** Tabulates entered laboratory indices alongside reference ranges and out-of-range directional flags.
3. **Cytomorphology Smear Imagery:** Embeds a high-resolution thumbnail of the blood smear with overlaid YOLOv8 bounding boxes (Red = RBC, Blue = WBC, Green = Platelet).
4. **Blood Cell Enumeration Table:** Displays absolute counts and cell density classifications.
5. **Leukocyte Differential Profile:** Renders a Matplotlib-generated pie chart of the 5-part WBC differential.
6. **Multimodal Anemia Assessment:** Displays final fused prediction, numerical confidence, active fusion strategy badge, and model version.
7. **Mandatory Clinical Disclaimer Banner:** High-visibility border detailing that the document is computer-generated for research/assistive screening and requires certified clinical sign-off.

---

## 21. Comprehensive Automated Test Suite & QA Verification

System stability, tensor dimensions, and routing behaviors are verified through an automated test suite across 6 dedicated test modules.

### 21.1 Test Suite Inventory & Results

The complete test suite was executed using Python's `unittest` framework:
```text
PS C:\Users\HP\Desktop\Blood_report_analyzer-main> .\.venv\Scripts\python -m unittest discover -s tests -p "test_*.py"
..........................................
----------------------------------------------------------------------
Ran 42 tests in 16.273s

OK
```

| Test Module | Functional Scope & Coverage | Tests | Result |
|---|---|:---:|:---:|
| [`tests/test_cell_detection.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_cell_detection.py) | YOLOv8 model loading, RGB/BGR channel ordering validation, bounding box format consistency, and tiled inference fallback on large images. | 3 | **PASS** |
| [`tests/test_anemia_image_model.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_anemia_image_model.py) | MobileNetV3-Small weight loading, image preprocessing transformations, softmax normalization, and missing-model error handling. | 2 | **PASS** |
| [`tests/test_anemia_model.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_anemia_model.py) | CBC Random Forest pipeline, feature scaling validation, input range boundary assertions, and deterministic inference. | 8 | **PASS** |
| [`tests/test_anemia_fusion.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_anemia_fusion.py) | Stacking Ensemble loading, 12-d feature engineering, average fallback logic, single-modality routing, and insufficient-data handling. | 6 | **PASS** |
| [`tests/test_gmu_architecture.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_gmu_architecture.py) | PyTorch GMU forward pass tensor dimensions ($B \times 128$), pre-extracted feature compatibility, missing-modality mask enforcement, and gradient backpropagation. | 17 | **PASS** |
| [`tests/test_gmu_integration.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_gmu_integration.py) | End-to-end integration of `predict_gmu_experimental()`, dual-modality inference, single-modality masks, scaler loading, and API schema adherence. | 6 | **PASS** |
| **Total Automated Tests** | **Full System Verification Coverage** | **42** | **100% PASS** |

---

## 22. Comprehensive Metric Reconciliation & Traceability Master Table

The table below reconciles every metric value documented across project history, linking each number to its exact physical checkpoint, split manifest, and evaluation script:

| Model & Task | Checkpoint Artifact Path | Evaluation Split / Sample Count | Reported Metric | Recomputed & Verified Metric | Historical Notes & Root Cause Analysis |
|---|---|---|:---:|:---:|---|
| **YOLOv8 Blood Cell Detector** | `models/yolov8_blood/best.pt` | BCCD Held-Out Test (56 images, 746 instances) | $\text{mAP}_{50} = 89.14\%$ | **89.14%** | Evaluated in `models/yolov8_blood/metrics.json`. Class mAP50: Platelet 82.74%, RBC 85.61%, WBC 99.06%. |
| **Original Baseline Smear Model** | `mobilenetv3_anemia_original_baseline.pth` | Unpurged 108 test images (88 Anemic, 20 Normal) | $\text{ROC-AUC} = 0.7477$ | **0.7477** | Unweighted baseline in `docs/ANEMIA_IMAGE_MODEL.md`. Collapsed specificity (5.0%) due to unweighted loss. |
| **Baseline Re-evaluation (Purged)** | `mobilenetv3_anemia_original_baseline.pth` | Purged 128 test images (108 Anemic, 20 Normal) | $\text{ROC-AUC} = 0.8755$ | **0.8752** | Baseline evaluated on cleaned split with cross-class duplicates purged (`independent_verification.json`). |
| **Weighted Image Retraining** | `mobilenetv3_anemia.pth` | Purged 128 test images (108 Anemic, 20 Normal) | $\text{ROC-AUC} = 0.8500$ | **0.8500** | Retrained with inverse class weights. Value "0.8148" in early summary was a clerical mistranscription of $88/108$. |
| **Cluster-Remediated Image Model** | `mobilenetv3_anemia_cluster_split.pth` | Remediated 129 test images (811 atomic clusters) | $\text{ROC-AUC} = 0.8167$ | **0.8167** | Zero cross-split duplicate leakage. Accuracy: 75.19%, Sensitivity: 77.27%, Specificity: 63.16% at $\tau^* = 0.5007$. |
| **CBC Model (Standard / Leaked)** | `models/anemia/anemia_model.joblib` | Held-Out Test (214 records, Kaggle) | $\text{Accuracy} = 100.0\%$ | **100.0%** | Target leakage: label derived from Hb. Hemoglobin feature importance is 83.32%. |
| **CBC Model (Master Dataset)** | `models/anemia/master_anemia_model.joblib`| Held-Out Test (809 records, Master CBC) | $\text{Accuracy} = 99.51\%$ | **99.51%** | Evaluated with Hb on 2,695-record master dataset. ROC-AUC: 0.9999. |
| **CBC Model (Screening / No Hb)**| Retrained Random Forest without Hb | Held-Out Test (809 records, Master CBC) | $\text{Accuracy} = 83.56\%$ | **83.56%** | Evaluated on `[Gender, MCV, MCH, MCHC]`. Sensitivity: 89.28%, Specificity: 77.11%, ROC-AUC: 0.8974. |
| **Production Stacking Ensemble** | `models/anemia_fusion/fusion_model.joblib` | Held-Out Test (360 synthetic combinations) | $\text{Accuracy} = 99.44\%$ | **99.44%** | Level-0: HistGB + ExtraTrees + MLP; Level-1: LogisticRegression. ROC-AUC: 0.9999. |
| **Experimental GMU Fusion** | `models/anemia_fusion/gmu_fusion_experimental.pth` | Held-Out Test (400 synthetic combinations) | $\text{Accuracy} = 97.50\%$ | **97.50%** | PyTorch GMU evaluated on synthetic test pairs. Sensitivity: 97.00%, Specificity: 98.00%, ROC-AUC: 0.9978. |

---

## 23. Root Cause Analysis of Resolved Engineering Defects

During system development and auditing, four major engineering defects were uncovered and resolved:

### 23.1 YOLOv8 RGB vs BGR Channel Inversion
- **Symptom:** Blood smears with obvious red erythrocytes were outputting near-zero RBC counts while reporting hundreds of false WBC detections.
- **Root Cause:** OpenCV and NumPy pipelines read images in RGB order when using PIL. However, YOLOv8 internally expects BGR numpy arrays. The swapped red and blue channels made red cells appear blue/purple, causing the model to classify them as leukocytes.
- **Resolution:** In [`src/detect.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/detect.py), explicit format checking was added: PIL Images are passed directly, and NumPy arrays are converted via `cv2.cvtColor(img, cv2.COLOR_RGB2BGR)` prior to inference.

### 23.2 Image Classifier Specificity Collapse (5.0%)
- **Symptom:** The initial MobileNetV3 image model flagged almost every normal blood smear as anemic (19 FP out of 20 Normal images).
- **Root Cause:** The training dataset had a 5.7:1 class imbalance (713 Anemic vs 124 Normal), and the model was trained with unweighted cross-entropy loss under a hardcoded $0.5$ threshold.
- **Resolution:** Implemented inverse-frequency class weighting ($w_0 = 3.39, w_1 = 0.59$) and calibrated the operating threshold to $\tau^* = 0.5007$ using Youden's J statistic, restoring specificity to 63.16%.

### 23.3 Cross-Split Field-of-View Leakage
- **Symptom:** Unclustered image splitting assigned adjacent microscopic fields from the same glass slide to both training and test sets.
- **Root Cause:** Random image-level shuffling ignored physical slide origin, allowing deep networks to memorize slide-specific staining tones.
- **Resolution:** Constructed an automated perceptual grouping pipeline that clustered the 837 images into 811 atomic clusters, guaranteeing zero cross-split cluster contamination.

### 23.4 Circular Target Leakage in CBC Models
- **Symptom:** The CBC Random Forest achieved 100% accuracy, creating a false impression of perfect diagnostic capability.
- **Root Cause:** Ground truth labels were derived from WHO Hemoglobin cutoffs, while raw `Hemoglobin` was fed as an input feature (83.3% feature importance).
- **Resolution:** Formally documented target leakage in all reports and conducted a leakage-free screening experiment excluding Hemoglobin to establish true baseline performance (83.56% accuracy).

---

## 24. Threats to Validity & System Limitations

To uphold medical research standards, all threats to system validity are transparently documented:

1. **Absence of Patient-Linked Multimodal Data:**
   - The primary scientific limitation of this project is that the blood smear images and CBC records originate from different patient cohorts with zero linking identifiers.
   - All multimodal fusion training (both the Stacking Ensemble and the GMU) was conducted on **synthetic combinations**.
   - Consequently, the high synthetic test accuracies (99.44% and 97.50%) **do not demonstrate clinical diagnostic efficacy on real patients**.
2. **Absence of Patient IDs in Smear Datasets:**
   - While the 837 smear images were clustered into 811 atomic clusters to eliminate field-of-view leakage, the underlying repositories provide no patient identifiers. Cluster independence does not guarantee patient independence.
3. **Staining & Optical Acquisition Variability:**
   - Peripheral blood smears are sensitive to staining protocols (Wright-Giemsa vs Leishman), stain precipitate artifacts, illumination color temperatures, and microscope camera sensor profiles. Cross-institutional generalization has not been prospectively validated.
4. **Target Leakage in Standard CBC Inputs:**
   - In clinical settings where full automated CBC panels are already available, machine learning prediction of anemia from Hemoglobin is clinically redundant.
5. **CPU Latency Constraints:**
   - On commodity dual-core CPUs without dedicated GPU hardware, running YOLOv8 detection, MobileNetV3 feature extraction, and PDF compilation requires 3 to 6 seconds per sample.

---

## 25. Ethical Safeguards, Medical Law & Regulatory Governance

### 25.1 Regulatory Device Classification

Under the regulatory frameworks of the **United States Food and Drug Administration (FDA)** and the **European Union Medical Device Regulation (EU MDR 2017/745)**:
- HemaVision AI is classified as **Software as a Medical Device (SaMD)** under IMDRF Risk Category **Class IIa (Moderate Risk)**.
- It is designed strictly as a **Clinical Decision Support (CDS)** tool under Section 520(o)(1)(E) of the US 21st Century Cures Act.
- To maintain CDS status and avoid classification as an unapproved Class III medical device, the software:
  1. Does **not** autonomously diagnose disease or dictate therapy.
  2. Presents intermediate morphological features, cell counts, and probability distributions so the clinician can independently review the basis of the recommendation.
  3. Enforces persistent visual disclaimers stating that the output must be confirmed by a licensed pathologist.

### 25.2 Patient Privacy & Data Security

- The local architecture processes all data on-premise without transmitting patient identifiable health information (PHI) to third-party cloud services.
- The SQLite database and local image stores do not retain patient names, national identity numbers, or social security numbers.
- Complies with HIPAA Privacy Rule de-identification standards (45 CFR § 164.514).

---

## 26. Future Clinical Roadmap & Research Directives

1. **Prospective Clinical Cohort Acquisition:**
   - Partner with university teaching hospitals to prospectively collect 500+ patient samples with **verified simultaneous collection** of high-magnification digital smear images and five-part automated CBC analyzer data from the same venipuncture draw.
2. **True Patient-Linked Multimodal Retraining:**
   - Retrain the Gated Multimodal Unit on the prospectively linked cohort to evaluate whether true multimodal fusion provides statistically significant diagnostic lift over single-modality models.
3. **Multi-Class Anemia Etiology Classification:**
   - Extend the classification head beyond binary anemia to differentiate specific underlying etiologies: Iron Deficiency Anemia, $\beta$-Thalassemia trait, Megaloblastic Anemia, Hemolytic Anemia, and Aplastic Anemia.
4. **Mobile & Edge Deployment:**
   - Quantize YOLOv8 and MobileNetV3 to 8-bit integers (INT8) via ONNX Runtime / TensorRT for deployment on low-cost Android smartphones coupled with 3D-printed microscope lens attachments.

---

## 27. User Manual & Step-by-Step Execution Guide

### 27.1 System Requirements

- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS 12+
- **Python:** Version 3.10, 3.11, or 3.12
- **Node.js (Optional, for React UI):** Version 18.0+
- **Hardware:** Minimum 4-core CPU, 8 GB RAM (NVIDIA GPU with CUDA recommended for fast inference)

### 27.2 Running the Application

1. **Activate Python Virtual Environment:**
   ```powershell
   # On Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Launch FastAPI Backend Service:**
   ```powershell
   python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   *Backend API Documentation:* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

3. **Launch Streamlit Frontend Application:**
   ```powershell
   python -m streamlit run frontend/app.py --server.port 8501
   ```
   *Streamlit Interface:* [http://localhost:8501](http://localhost:8501)

4. **Launch React / TanStack Web Interface (Optional):**
   ```powershell
   cd ui
   npm run dev
   ```
   *React Interface:* [http://localhost:8080](http://localhost:8080)

### 27.3 Running the Automated Test Suite

To run all 42 automated unit and integration tests:
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

### 27.4 Reproducing Model Training Pipelines

- **Train YOLOv8 Blood Cell Detector:**
  ```powershell
  python scripts/train_blood_detector.py --epochs 3 --batch 16 --imgsz 320
  ```
- **Train Cluster-Remediated Image Classifier:**
  ```powershell
  python scripts/train_anemia_image_cluster_split.py --epochs 5 --batch 16
  ```
- **Train CBC Models & Run Leakage Analysis:**
  ```powershell
  python scripts/train_anemia_model.py
  python scripts/audit_cbc_leakage_experiment.py
  ```
- **Train Production Stacking Ensemble:**
  ```powershell
  python scripts/train_multimodal_stacking.py
  ```
- **Train Experimental Gated Multimodal Unit (GMU):**
  ```powershell
  python scripts/train_gmu_fusion.py --epochs 10 --batch_size 32 --lr 0.001
  ```

---

## 28. Appendix A: Mathematical Derivations & Gating Proofs

### 28.1 Dynamic GMU Gate Behavior Under Asymmetric Loss Gradients

Let the total loss be binary cross-entropy:
$$\mathcal{L} = - \left[ y \log \hat{y} + (1 - y) \log (1 - \hat{y}) \right]$$
where $\hat{y} = \sigma(W_h h_{\text{fused}} + b_h)$ and $h_{\text{fused}} = z \odot h_{\text{img}} + (\mathbf{1} - z) \odot h_{\text{cbc}}$.

By the chain rule, the gradient of the loss with respect to the gate pre-activation vector $u_z = W_z [h_{\text{img}}, h_{\text{cbc}}] + b_z$ is:
$$\frac{\partial \mathcal{L}}{\partial u_z} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h_{\text{fused}}} \cdot \frac{\partial h_{\text{fused}}}{\partial z} \cdot \frac{\partial z}{\partial u_z}$$

1. $\frac{\partial \mathcal{L}}{\partial \hat{y}} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}$
2. $\frac{\partial \hat{y}}{\partial h_{\text{fused}}} = \hat{y}(1 - \hat{y}) W_h^T$
3. $\frac{\partial h_{\text{fused}}}{\partial z} = \text{diag}(h_{\text{img}} - h_{\text{cbc}})$
4. $\frac{\partial z}{\partial u_z} = \text{diag}(z \odot (\mathbf{1} - z))$

Multiplying these terms yields:
$$\frac{\partial \mathcal{L}}{\partial u_z} = (\hat{y} - y) \cdot \left[ W_h^T \odot (h_{\text{img}} - h_{\text{cbc}}) \odot z \odot (\mathbf{1} - z) \right]$$

#### Mathematical Insight:
- When candidate representation $h_{\text{img}}$ aligns with the true gradient direction better than $h_{\text{cbc}}$ (i.e., $h_{\text{img}} - h_{\text{cbc}}$ reduces error), the gradient pushes $u_z$ upward, increasing gate activation $z \to 1$.
- Conversely, when CBC features are more discriminative, $z \to 0$, routing neural capacity toward laboratory parameters.

---

## 29. Appendix B: Database Schema & Entity Relationships

```
┌────────────────────────────────────────────────────────────────────────┐
│                        analyses (SQLite Table)                         │
├────────────────────────────────┬─────────────────┬─────────────────────┤
│ Column Name                    │ SQLite Type     │ Description         │
├────────────────────────────────┼─────────────────┼─────────────────────┤
│ id                             │ INTEGER         │ Primary Key         │
│ analysis_id                    │ VARCHAR(64)     │ Unique GUID String  │
│ timestamp                      │ DATETIME        │ Execution timestamp │
│ image_path                     │ VARCHAR(255)    │ Source image path   │
│ annotated_image_path           │ VARCHAR(255)    │ YOLO annotated path │
│ rbc_count                      │ INTEGER         │ YOLO RBC detections │
│ wbc_count                      │ INTEGER         │ YOLO WBC detections │
│ platelet_count                 │ INTEGER         │ YOLO PLT detections │
│ wbc_differential_json          │ TEXT (JSON)     │ Subtype percentages │
│ image_anemia_prediction        │ VARCHAR(32)     │ Normal / Anemic     │
│ image_anemia_probability       │ FLOAT           │ MobileNetV3 P(img)  │
│ cbc_anemia_prediction          │ VARCHAR(32)     │ Normal / Anemic     │
│ cbc_anemia_probability         │ FLOAT           │ Random Forest P(cbc)│
│ cbc_inputs_json                │ TEXT (JSON)     │ Serialized lab params│
│ final_prediction               │ VARCHAR(32)     │ Fused verdict       │
│ final_probability              │ FLOAT           │ Fused probability   │
│ fusion_strategy                │ VARCHAR(64)     │ Active fusion mode  │
│ pdf_report_path                │ VARCHAR(255)    │ Output PDF path     │
└────────────────────────────────┴─────────────────┴─────────────────────┘
```

---

## 30. Appendix C: Complete REST API Endpoint Inventory

| HTTP Method | Route Path | Request Content-Type | Response Schema | Description |
|---|---|---|---|---|
| `POST` | `/api/analyze` | `multipart/form-data` | `AnalysisResponse` | Full pipeline inference for smear image, CBC parameters, or both. |
| `GET` | `/api/history` | `None` | `List[AnalysisSummary]` | Paginated history of prior analyses. |
| `GET` | `/api/history/{analysis_id}` | `None` | `AnalysisDetail` | Full serialized record and parameters for single analysis. |
| `GET` | `/api/reports/{filename}` | `None` | `application/pdf` | Streams generated clinical PDF report. |
| `GET` | `/api/health` | `None` | `HealthCheckResponse` | System health check and model file verification. |

---

## 31. Appendix D: Checkpoint Inventory & File Hash Registry

| Model Checkpoint Artifact | Relative Path | Size | Architecture / Task |
|---|---|:---:|---|
| YOLOv8 Blood Detector | `models/yolov8_blood/best.pt` | 6.0 MB | YOLOv8n (3 classes: RBC, WBC, Platelet) |
| Cluster-Remediated Image Model | `models/anemia_image_cluster_split/mobilenetv3_anemia_cluster_split.pth` | 3.8 MB | MobileNetV3-Small (Binary Anemia) |
| Baseline Image Model (Preserved) | `models/anemia_image/mobilenetv3_anemia_original_baseline.pth` | 3.8 MB | MobileNetV3-Small (Unweighted) |
| CBC Random Forest Classifier | `models/anemia/anemia_model.joblib` | 1.2 MB | Random Forest (5 CBC features) |
| Production Stacking Ensemble | `models/anemia_fusion/fusion_model.joblib` | 2.4 MB | HistGB + ExtraTrees + MLP Meta-Classifier |
| Experimental GMU Checkpoint | `models/anemia_fusion/gmu_fusion_experimental.pth` | 2.1 MB | PyTorch Gated Multimodal Unit ($d=128$) |
| Experimental GMU Scaler | `models/anemia_fusion/gmu_experimental_scaler.joblib` | 1.8 KB | StandardScaler fitted on Train CBCs |

---

## 32. Appendix E: Software Dependencies & Environment Manifest

Extracted directly from project environment configuration:
- `python = ">=3.10,<3.13"`
- `torch = "2.1.0"`
- `torchvision = "0.16.0"`
- `ultralytics = "8.1.0"`
- `opencv-python = "4.9.0.80"`
- `scikit-learn = "1.9.0"`
- `fastapi = "0.110.0"`
- `uvicorn = "0.29.0"`
- `pydantic = "2.6.4"`
- `sqlalchemy = "2.0.28"`
- `streamlit = "1.32.0"`
- `fpdf2 = "2.8.8"`
- `matplotlib = "3.8.0"`
- `pandas = "2.2.1"`
- `numpy = "1.26.4"`

---

## 33. References & Academic Citations

1. **World Health Organization (WHO):** *Haemoglobin concentrations for the diagnosis of anaemia and assessment of severity.* Vitamin and Mineral Nutrition Information System. Geneva, World Health Organization, 2011 (WHO/NMH/NHD/MNM/11.1).
2. **Arevalo, J., Solorio, T., Montes-y-Gómez, M., & González, F. A. (2017):** *Gated Multimodal Units for Multimodal Fusion.* International Conference on Learning Representations (ICLR) Workshops.
3. **Jocher, G., Chaurasia, A., & Qiu, J. (2023):** *Ultralytics YOLOv8.* Version 8.1.0. GitHub repository: `https://github.com/ultralytics/ultralytics`.
4. **Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q. V., & Adam, H. (2019):** *Searching for MobileNetV3.* IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1314–1324.
5. **Shenggan (2019):** *BCCD (Blood Cell Count and Detection) Dataset.* MIT License. GitHub repository: `https://github.com/Shenggan/BCCD_Dataset`.
6. **Mendeley Data (2020):** *AneRBC: Peripheral Blood Smear Dataset for Anemia Classification.* Mendeley Data, V1, DOI: `10.17632/hms3sjzt7f.1`.
7. **Chulalongkorn University (2022):** *Chula-RBC-12: A Dataset of Red Blood Cell Microscopy Images with 12 Classes of Morphological Alterations.* Biomedical Informatics and Computer Vision Laboratory.
8. **Pedregosa, F., et al. (2011):** *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, pp. 2825–2830.
9. **Youden, W. J. (1950):** *Index for rating diagnostic tests.* Cancer, 3(1), pp. 32–35.
10. **US Food and Drug Administration (FDA) (2022):** *Clinical Decision Support Software: Guidance for Industry and Food and Drug Administration Staff.* Center for Devices and Radiological Health (CDRH).
