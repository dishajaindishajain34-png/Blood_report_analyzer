# AI-Based Smart Blood Report Analyzer 🩸

An AI-powered system that analyzes blood smear microscope images to automatically detect, classify, and count blood cells — generating a downloadable clinical-grade PDF report in seconds.

---

## Features

| Feature | Status |
|---|---|
| 🎯 YOLOv8 cell detection (RBC / WBC / Platelet) | ✅ Completed |
| 🔬 OpenCV preprocessing (CLAHE + Gaussian) | ✅ Completed |
| 🧬 WBC differential count (5 subtypes) | ✅ Completed (PyTorch ResNet50 / Sklearn) |
| 📊 WHO normal range & Anemia flagging | ✅ Completed |
| 📄 PDF report generation (FPDF2 + Charts) | ✅ Completed |
| 🌐 Streamlit frontend (Interactive UI) | ✅ Completed |
| ⚡ FastAPI REST backend (Swagger / Docs) | ✅ Completed |
| 🤖 ResNet50 / Sklearn WBC classifier | ✅ Completed |
| 🗄️ SQLite patient record storage & History | ✅ Completed |

---

## Quick Start (Demo — Phase 1)

### 1. Install dependencies

```bash
cd Blood_report_analyzer
pip install -r requirements.txt
```

### 2. Run the Streamlit app

```bash
streamlit run frontend/app.py
```

This opens the full UI in your browser. Upload any blood smear image (JPEG/PNG) and the system will:
- Preprocess with CLAHE contrast enhancement
- Detect RBCs, WBCs, and Platelets using YOLOv8 (auto-downloads the pre-trained BCCD model on first run)
- Run WBC differential estimation
- Check WHO reference ranges and flag abnormalities
- Generate a downloadable PDF report

> **Note:** The YOLOv8 detection weights are downloaded automatically on first run. By default the project prefers a local `models/yolov8_blood/best.pt` if present, otherwise it will download the published weights from the remote repository (HuggingFace/Ultralytics URL). See `src/detect.py` for the source URL.

Model filenames used by this repo (default paths under `models/`):

- `models/yolov8_blood/best.pt` — YOLOv8 detector weights (auto-downloaded if missing)
- `models/resnet50_wbc/wbc_classifier_sklearn.pth` — sklearn ensemble WBC classifier (joblib bytes inside .pth)

Migration note: schema additions (nullable anemia columns and timezone-aware `created_at`) were added — see MIGRATION.md for upgrade guidance.

If you run the backend API behind a browser-based frontend, set the `FRONTEND_ORIGIN` environment variable to your frontend origin (default: `http://127.0.0.1:8501`). See `backend/main.py` for details.

### 3. Run the FastAPI backend (optional)

```bash
# Using uvicorn (recommended):
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# Or use the module runner if `uvicorn` is not on your PATH but installed in the venv:
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

API docs available at: http://127.0.0.1:8000/docs

---

## Project Structure

```
Blood_report_analyzer/
├── src/
│   ├── preprocess.py        # OpenCV CLAHE preprocessing
│   ├── detect.py            # YOLOv8 cell detection
│   ├── classify_wbc.py      # ResNet50 WBC subtype classifier
│   ├── normal_ranges.py     # WHO reference range flagging
│   ├── report_generator.py  # FPDF2 PDF report builder
│   └── utils.py             # Shared utilities
├── backend/
│   ├── main.py              # FastAPI application
│   ├── schemas.py           # Pydantic models
│   └── routes/
│       ├── analyze.py       # POST /analyze
│       └── report.py        # GET /report/{id}
├── frontend/
│   └── app.py               # Streamlit UI
├── training/
│   ├── train_yolov8.py      # YOLOv8 fine-tuning
│   ├── train_resnet.py      # ResNet50 fine-tuning
│   └── data.yaml            # BCCD dataset config
├── models/                  # Model weights (gitignored)
├── data/                    # Datasets and sample images
├── reports/                 # Generated PDF reports
└── requirements.txt
```

---

## Training Your Own Models

### YOLOv8 (Blood Cell Detection)

```bash
# With Roboflow BCCD dataset download
python training/train_yolov8.py --use-roboflow --roboflow-key YOUR_KEY --epochs 50

# With local dataset (update training/data.yaml first)
python training/train_yolov8.py --epochs 50 --model yolov8s
```

### ResNet50 (WBC Subtype Classifier) — Phase 2

```bash
# Requires WBC subtype dataset in data/wbc_classifier/ (ImageFolder format)
python training/train_resnet.py --epochs 30 --batch 32
```

---

## Dataset Sources

| Dataset | Purpose | Link |
|---|---|---|
| BCCD Dataset | RBC/WBC/Platelet detection (YOLOv8) | [GitHub](https://github.com/Shenggan/BCCD_Dataset) |
| Blood Cell Count (Kaggle) | Additional training images | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) |
| WBC Differential (Roboflow) | WBC subtype classification | [Roboflow Universe](https://universe.roboflow.com/) |

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/analyze` | POST | Upload image → full analysis JSON |
| `/report/{id}` | GET | Download PDF report |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |

---

## Tech Stack

| Component | Technology |
|---|---|
| Cell Detection | YOLOv8 (Ultralytics) |
| WBC Classification | ResNet50 (PyTorch) + sklearn ensemble fallback |
| Preprocessing | OpenCV + CLAHE |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| PDF Reports | FPDF2 + Matplotlib |
| Reference Ranges | WHO Standard Ranges |

---

## Disclaimer

> This system is intended as a **preliminary AI-assisted tool only**. All outputs must be reviewed by a certified pathologist or licensed medical professional. This software does not constitute a medical diagnosis and should not be used as a substitute for professional medical advice.

---

## Development Phases

- **Phase 1** (Weeks 1–3): Working demo — YOLOv8 detection, estimated WBC differential, PDF reports ✅
- **Phase 2** (Weeks 4–8): ResNet50 WBC classifier, SQLite records, full FastAPI backend
- **Phase 3** (Weeks 9–12): Docker, Supabase, evaluation metrics, production deployment



**** make two terminals and do run

for frontend to run
<!-- (.venv) PS C:\Users\Admin\Downloads\Blood_report_analyzer\ui> npm run dev -- --host 0.0.0.0 --port 8083  -->


for backend to run
(.venv) PS C:\Users\Admin\Downloads\Blood_report_analyzer> cd "C:\Users\Admin\Downloads\Blood_report_analyzer"
>> .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
<!-- (.venv) PS C:\Users\Admin\Downloads\Blood_report_analyzer> python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload -->
<!-- inorder to terminate the running localhost /// (.venv) PS C:\Users\Admin\Downloads\Blood_report_analyzer> .\.venv\Scripts\Activate.ps1   -->