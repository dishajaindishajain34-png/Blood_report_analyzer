# AI Blood Report Analyzer — User Guide

## How to Run

Open a terminal in the project folder and run:

```bash
cd C:\Users\HP\Desktop\Blood_report_analyzer
streamlit run frontend/app.py
```

Then open your browser at **http://localhost:8501**

That's it. The app starts in ~5 seconds and is fully self-contained (no internet needed after first run).

> [!TIP]
> To also run the REST API backend (for integrations):
> ```bash
> uvicorn backend.main:app --reload --port 8000
> ```
> Then visit **http://localhost:8000/docs** for the interactive API explorer.

---

## All Features — Tab by Tab

### 🔬 Tab 1: Analyze

This is the main analysis screen. Here's the full flow:

#### Step 1 — Choose your input mode
| Mode | Use when |
|---|---|
| **Single Image** | You have one blood smear photo |
| **Batch (multiple images)** | You have 2–10 images of the same sample from different fields of view — results get averaged for better accuracy |

#### Step 2 — Load an image
- **Upload**: Drag & drop or click to upload a JPEG/PNG blood smear image
- **Use Sample Image**: Loads a real BCCD dataset blood smear built into the project — great for demo

After loading, a thumbnail preview appears so you can confirm the right image is selected.

#### Step 3 — Optional: Set Patient Info (sidebar)
Fill in Patient Name, Patient ID, and Age in the left sidebar. These get saved to the database with the result — useful for tracking patients over time.

#### Step 4 — Adjust detection settings (sidebar)
| Setting | Default | Effect |
|---|---|---|
| **Detection Confidence** | 0.35 | Lower = detects more cells (may include false positives). Raise for stricter detection |
| **NMS IOU Threshold** | 0.40 | Controls overlapping box removal. Lower = fewer merged detections |

#### Step 5 — Click ▶ Run Analysis
The pipeline runs through 5 steps (shown as a progress bar):
1. **Preprocessing** — CLAHE contrast enhancement + Gaussian blur
2. **YOLOv8 Detection** — neural network draws bounding boxes around every RBC, WBC, and platelet
3. **WBC Classification** — sklearn model classifies each white blood cell crop into a subtype
4. **Calibration** — raw frame counts scaled to physiological cells/µL estimates
5. **Report Generation** — PDF compiled with all results

#### Results You Get

**Analysis ID** — unique code like `BRA-20260502115345-076220` for every report

**Overall Result badge**
- 🟢 **Normal** — all counts within WHO reference ranges
- 🔴 **Abnormal** — one or more values outside normal ranges

**Side-by-side images**
- Left: your original blood smear
- Right: YOLOv8 annotated image with coloured bounding boxes
  - 🔵 Cyan boxes = RBC (Red Blood Cells)
  - ⬜ White boxes = WBC (White Blood Cells)
  - Confidence score shown next to each box (e.g., `WBC 0.84`)

**Cell Count Cards** (calibrated to cells/µL)
| Cell | Normal Range |
|---|---|
| RBC | 4.2M – 6.1M /µL |
| WBC | 4,000 – 11,000 /µL |
| Platelet | 150,000 – 400,000 /µL |

**WBC Differential Donut Chart** — shows the breakdown of white blood cell subtypes:
- Neutrophil (normal: 50–70%)
- Lymphocyte (normal: 20–40%)
- Monocyte (normal: 2–8%)
- Eosinophil (normal: 1–4%)
- Basophil (normal: 0–1%)

> [!NOTE]
> Currently shows **"Sklearn"** in the donut center — the sklearn classifier is active. If it says **"Est."** it means placeholder mode (no model loaded).

**Clinical Findings flags**
Each flag explains what's abnormal and what it might indicate:
- `⚠ White Blood Cells: 25,666 /µL (High) — Possible Infection / Leukocytosis`
- `⚠ WBC Differential — Neutrophil: 100.0% (High) — Neutrophilia`

**📄 Download PDF Report** — generates a clinical-style PDF containing:
- Patient info, analysis ID, timestamp
- Annotated blood smear image
- Cell count table with WHO reference ranges
- WBC differential table
- All clinical flags with explanations
- Medical disclaimer

---

### 📋 Tab 2: History

Every analysis run is automatically saved to a local SQLite database.

**Stats row at the top:**
- Total Analyses run
- How many were Normal vs Abnormal
- Abnormal percentage

**Search box** — filter records by patient name or patient ID

**Records table** — sortable table showing:
- Analysis ID, Patient, Age, Date/Time
- Calibrated RBC / WBC / Platelet counts
- Detection confidence
- Normal / Abnormal result

**WBC Trend chart** — if you have 3+ records, a line chart appears showing WBC and RBC trends over time — useful for monitoring a patient across multiple visits.

---

### ℹ️ Tab 3: About

Shows the full tech stack, dataset info, model status, and the Phase 3 roadmap.

---

## How the Pipeline Works Internally

```
Your Image
    │
    ▼
┌─────────────────────────────────┐
│  PREPROCESSING (src/preprocess) │
│  • Resize to 640×640            │
│  • CLAHE contrast enhancement   │
│  • Gaussian blur (noise reduce) │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  YOLOV8 DETECTION (src/detect)  │
│  • Model: YOLOv8 detector weights (`models/yolov8_blood/best.pt`) — auto-downloaded from the configured remote if missing
│  • Detects: RBC / WBC / Platelet│
│  • Outputs bounding boxes +     │
│    confidence scores            │
│  • Draws annotated image        │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  WBC CROP EXTRACTION            │
│  • Crops each WBC bounding box  │
│  • Passes crops to classifier   │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  WBC CLASSIFIER (classify_wbc)  │
│  • Primary: sklearn SVM + RF ensemble stored as `models/resnet50_wbc/wbc_classifier_sklearn.pth`
│  • Features: HOG + color hist + HSV histograms
│  • Classifies each crop into one of 5 WBC subtypes; if model file is missing the app uses a safe fallback estimation
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  CALIBRATION (src/utils)        │
│  • Scales frame counts to /µL   │
│  • RBC: ×960,000                │
│  • WBC: ×2,333                  │
│  • Platelet: ×25,000            │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  WHO FLAGGING (normal_ranges)   │
│  • Compares each count to       │
│    reference ranges             │
│  • Generates clinical flags     │
│  • Sets Overall = Normal /      │
│    Abnormal                     │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  PDF REPORT (report_generator)  │
│  • FPDF2 — Latin-1 safe         │
│  • Includes images, tables,     │
│    flags, disclaimer            │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  SQLITE DATABASE (database/)    │
│  • Saves record to              │
│    data/blood_analyzer.db       │
│  • Viewable in History tab      │
└─────────────────────────────────┘
```

---

## Key Files

```
Blood_report_analyzer/
├── frontend/app.py              ← Streamlit UI (run this)
├── backend/main.py              ← FastAPI REST backend
├── src/
│   ├── preprocess.py            ← CLAHE image preprocessing
│   ├── detect.py                ← YOLOv8 inference
│   ├── classify_wbc.py          ← sklearn WBC classifier
│   ├── normal_ranges.py         ← WHO reference range checker
│   ├── report_generator.py      ← PDF generation
│   ├── batch_analyze.py         ← Multi-image averaging
│   └── utils.py                 ← Calibration + shared helpers
├── database/
│   ├── models.py                ← SQLite schema (SQLAlchemy)
│   ├── crud.py                  ← Save/query records
│   └── connection.py            ← DB engine setup
├── models/
│   ├── yolov8_blood/best.pt     ← YOLOv8 weights (5.9 MB)
│   └── resnet50_wbc/
│       └── wbc_classifier_sklearn.pth   ← sklearn model (joblib, 2.9 MB)
├── data/
│   ├── sample_images/           ← Built-in demo images
│   ├── bccd_raw/                ← 364 BCCD training images
│   ├── wbc_classifier/          ← 372 WBC crop training set
│   └── blood_analyzer.db        ← Patient records database
└── training/
    ├── train_wbc_sklearn.py     ← Retrain WBC classifier
    ├── train_yolov8.py          ← Retrain YOLOv8 on BCCD
    └── extract_wbc_crops.py     ← Extract WBC crops from BCCD
```

---

## FAQ

**Q: Why does the WBC differential show 100% Neutrophil?**  
A: The WBC classifier was trained on pseudo-labeled crops (no true subtype labels exist in BCCD). The model learned Neutrophil as the dominant class. To fix this, download a properly labeled WBC subtype dataset and run `python training/train_wbc_sklearn.py` again.

**Q: Why is the result always "Abnormal"?**  
A: The sample BCCD image has 11 WBCs in a single high-magnification frame — that calibrates to ~25,000 WBC/µL which is genuinely above the normal range (4,000–11,000). This correctly flags Leukocytosis.

**Q: How accurate is the cell detection?**  
A: ~63% average confidence using the pre-trained YOLOv8n model. Running `python training/train_yolov8.py` fine-tunes it on the full BCCD dataset which should improve to ~85%+.

**Q: Where are generated PDFs saved?**  
A: In `reports/` folder as `BRA-YYYYMMDDHHMMSS-XXXXXX.pdf`

**Q: How do I retrain the WBC classifier?**  
```bash
python training/train_wbc_sklearn.py
```
Takes ~90 seconds. New model overwrites `models/resnet50_wbc/wbc_classifier_sklearn.pth`.
# POST /analyze  — upload image, get JSON result
# GET  /report/{analysis_id}  — download PDF
```
