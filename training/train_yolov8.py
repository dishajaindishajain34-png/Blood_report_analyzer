"""
train_yolov8.py — Fine-tune YOLOv8 on the BCCD blood cell detection dataset.

Usage:
    python training/train_yolov8.py [--epochs 50] [--model yolov8n] [--use-roboflow]

The script downloads the BCCD dataset via Roboflow (free API key required),
then fine-tunes YOLOv8 starting from a pre-trained COCO checkpoint.

Results are saved to models/yolov8_blood/best.pt.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, ensure_dir, project_path

log = get_logger("train_yolov8")


def download_bccd_roboflow(api_key: str, dataset_dir: Path) -> Path:
    """Download the BCCD dataset from Roboflow in YOLOv8 format."""
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError("Install roboflow: pip install roboflow")

    log.info("Downloading BCCD dataset from Roboflow …")
    rf       = Roboflow(api_key=api_key)
    project  = rf.workspace("roboflow-100").project("bccd-ouzjj")
    dataset  = project.version(1).download("yolov8", location=str(dataset_dir))
    data_yaml = Path(dataset.location) / "data.yaml"
    log.info("Dataset downloaded to: %s", dataset.location)
    return data_yaml


def train(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Install ultralytics: pip install ultralytics")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_dir = project_path("data", "bccd")
    ensure_dir(dataset_dir)

    if args.use_roboflow:
        api_key = args.roboflow_key or os.getenv("ROBOFLOW_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Roboflow API key required. Pass --roboflow-key or set ROBOFLOW_API_KEY env var."
            )
        data_yaml = download_bccd_roboflow(api_key, dataset_dir)
    else:
        # Use a manually placed data.yaml
        data_yaml = project_path("training", "data.yaml")
        if not data_yaml.exists():
            log.warning(
                "data.yaml not found at %s. "
                "Please either use --use-roboflow or manually place your BCCD dataset "
                "and training/data.yaml.", data_yaml
            )
            sys.exit(1)

    # ── Model ─────────────────────────────────────────────────────────────────
    base_model = f"{args.model}.pt"   # e.g. yolov8n.pt (auto-downloaded)
    log.info("Loading base model: %s", base_model)
    model = YOLO(base_model)

    # ── Training ──────────────────────────────────────────────────────────────
    out_dir = project_path("runs", "train")
    log.info("Starting training: epochs=%d, imgsz=%d, batch=%d",
             args.epochs, args.imgsz, args.batch)

    results = model.train(
        data        = str(data_yaml),
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        name        = "blood_cell_detector",
        project     = str(out_dir),
        patience    = 15,
        lr0         = 0.01,
        optimizer   = "AdamW",
        augment     = True,
        plots       = True,
        save        = True,
        device      = args.device,
    )

    # ── Copy best weights to models/ ──────────────────────────────────────────
    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = project_path("models", "yolov8_blood", "best.pt")
    ensure_dir(best_dst.parent)

    if best_src.exists():
        import shutil
        shutil.copy2(str(best_src), str(best_dst))
        log.info("Best weights saved → %s", best_dst)
    else:
        log.warning("best.pt not found at %s", best_src)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on BCCD blood cell dataset")
    parser.add_argument("--epochs",        type=int,   default=50,       help="Number of training epochs")
    parser.add_argument("--imgsz",         type=int,   default=640,      help="Input image size")
    parser.add_argument("--batch",         type=int,   default=16,       help="Batch size")
    parser.add_argument("--model",         type=str,   default="yolov8n",help="Base model (yolov8n/s/m/l)")
    parser.add_argument("--device",        type=str,   default="",       help="Device: '' (auto), 'cpu', '0' (GPU 0)")
    parser.add_argument("--use-roboflow",  action="store_true",          help="Download BCCD via Roboflow API")
    parser.add_argument("--roboflow-key",  type=str,   default="",       help="Roboflow API key")
    args = parser.parse_args()

    train(args)
