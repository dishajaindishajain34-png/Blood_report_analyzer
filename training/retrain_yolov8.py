"""
Retrain YOLOv8 on expanded BCCD dataset (with archive integration)
"""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, ensure_dir, project_path

log = get_logger("train_yolov8_retrain")


def train_yolov8():
    """Retrain YOLOv8 on expanded dataset"""
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed. Installing...")
        os.system(f"{sys.executable} -m pip install ultralytics")
        from ultralytics import YOLO

    # Configuration
    epochs = 20  # Reduced for CPU training (use 75+ for better accuracy if GPU available)
    imgsz = 640
    batch = 8  # Reduced batch size for CPU
    device = "cpu"  # Use CPU training
    
    # Paths
    data_yaml = project_path("training", "data.yaml")
    out_dir = project_path("runs", "train")
    base_model = "yolov8n.pt"
    
    log.info("=" * 70)
    log.info("YOLOv8 RETRAIN on Expanded BCCD Dataset")
    log.info("=" * 70)
    log.info(f"Data YAML: {data_yaml}")
    log.info(f"Epochs: {epochs} | Batch: {batch} | Image Size: {imgsz}")
    log.info(f"Device: {device} (CPU Training)")
    log.info("Note: CPU training will be slower but will work without GPU")
    log.info(f"Output directory: {out_dir}")
    
    # Verify data.yaml exists
    if not data_yaml.exists():
        log.error(f"data.yaml not found at {data_yaml}")
        sys.exit(1)
    
    # Load model
    log.info(f"\nLoading base model: {base_model}")
    model = YOLO(base_model)
    
    # Train
    log.info("\nStarting training...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="blood_cell_detector",
        project=str(out_dir),
        patience=15,
        lr0=0.01,
        optimizer="AdamW",
        augment=True,
        plots=True,
        save=True,
        device=device,
        verbose=True,
    )
    
    # Copy best weights
    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = project_path("models", "yolov8_blood", "best.pt")
    
    if best_src.exists():
        ensure_dir(best_dst.parent)
        import shutil
        shutil.copy2(str(best_src), str(best_dst))
        log.info(f"\n✓ Best weights saved → {best_dst}")
    else:
        log.warning(f"best.pt not found at {best_src}")
    
    log.info("\n" + "=" * 70)
    log.info("Training Complete!")
    log.info("=" * 70)
    
    return results


if __name__ == "__main__":
    train_yolov8()
