import sys
import json
import shutil
from pathlib import Path
import time

# Add project root to path
ROOT = Path("c:/Users/HP/Desktop/Blood_report_analyzer-main")
sys.path.insert(0, str(ROOT))

from src.utils import get_logger, ensure_dir

log = get_logger("retrain_and_evaluate")

def run():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics is not installed. Please run pip install ultralytics")
        sys.exit(1)
        
    old_model_path = ROOT / "models" / "yolov8_blood" / "best.pt"
    data_yaml_path = ROOT / "data" / "blood_cells" / "data.yaml"
    improved_model_dir = ROOT / "models" / "yolov8_blood"
    improved_model_path = improved_model_dir / "best_platelet_improved.pt"
    
    if not old_model_path.exists():
        print(f"[ERROR] Original model not found at {old_model_path}")
        sys.exit(1)
        
    if not data_yaml_path.exists():
        print(f"[ERROR] data.yaml not found at {data_yaml_path}")
        sys.exit(1)
        
    print("=" * 80)
    print("  BLOOD CELL DETECTOR - RETRAINING & EVALUATION PIPELINE")
    print("=" * 80)
    print(f"Data config: {data_yaml_path}")
    print(f"Original model: {old_model_path}")
    
    # ── 1. Evaluate Old Model on TEST Split ──
    print("\n[STEP 1] Evaluating original model on held-out test split...")
    old_model = YOLO(str(old_model_path))
    old_results = old_model.val(data=str(data_yaml_path), split="test", verbose=False)
    
    old_metrics = {}
    for idx, name in old_results.names.items():
        old_metrics[name] = {
            "precision": round(float(old_results.box.p[idx]), 4),
            "recall": round(float(old_results.box.r[idx]), 4),
            "map50": round(float(old_results.box.ap50[idx]), 4),
            "map50-95": round(float(old_results.box.ap[idx]), 4),
        }
    old_overall = {
        "precision": round(float(old_results.box.mp), 4),
        "recall": round(float(old_results.box.mr), 4),
        "map50": round(float(old_results.box.map50), 4),
        "map50-95": round(float(old_results.box.map), 4), # map50-95 is a scalar overall map
    }
    
    print("\nOriginal Model Test Metrics:")
    for name, m in old_metrics.items():
        print(f"  {name:<10}: Precision={m['precision']:.2%}, Recall={m['recall']:.2%}, mAP50={m['map50']:.2%}")
        
    # ── 2. Fine-tune Model on Combined Dataset ──
    print("\n[STEP 2] Fine-tuning model to improve Platelet detection on CPU...")
    # We fine-tune starting from the original model weights
    t0 = time.time()
    train_results = old_model.train(
        data=str(data_yaml_path),
        epochs=3,             # Short CPU-friendly training
        imgsz=320,            # 320 is 4x faster to train than 640 on CPU
        batch=16,
        name="blood_cell_detector_improved",
        project=str(ROOT / "runs" / "train"),
        device="cpu",
        patience=10,
        lr0=0.005,           # Lower learning rate to avoid destroying existing features
        optimizer="AdamW",
        augment=True,
        plots=True,
        save=True,
        verbose=False
    )
    t_train = time.time() - t0
    print(f"Training complete in {t_train:.1f}s.")
    
    # Copy best weights
    best_src = Path(train_results.save_dir) / "weights" / "best.pt"
    if best_src.exists():
        ensure_dir(improved_model_path.parent)
        shutil.copy2(str(best_src), str(improved_model_path))
        print(f"Saved improved model to: {improved_model_path}")
    else:
        print("[ERROR] Training output best.pt not found!")
        sys.exit(1)
        
    # ── 3. Evaluate Improved Model on TEST Split ──
    print("\n[STEP 3] Evaluating improved model on held-out test split...")
    new_model = YOLO(str(improved_model_path))
    new_results = new_model.val(data=str(data_yaml_path), split="test", verbose=False)
    
    new_metrics = {}
    for idx, name in new_results.names.items():
        new_metrics[name] = {
            "precision": round(float(new_results.box.p[idx]), 4),
            "recall": round(float(new_results.box.r[idx]), 4),
            "map50": round(float(new_results.box.ap50[idx]), 4),
            "map50-95": round(float(new_results.box.ap[idx]), 4),
        }
    new_overall = {
        "precision": round(float(new_results.box.mp), 4),
        "recall": round(float(new_results.box.mr), 4),
        "map50": round(float(new_results.box.map50), 4),
        "map50-95": round(float(new_results.box.map), 4),
    }
    
    print("\nImproved Model Test Metrics:")
    for name, m in new_metrics.items():
        print(f"  {name:<10}: Precision={m['precision']:.2%}, Recall={m['recall']:.2%}, mAP50={m['map50']:.2%}")
        
    # ── 4. Compare and Save Metrics ──
    print("\n[STEP 4] Generating Old vs New Model Comparison:")
    print("=" * 95)
    print(f"{'Class':<12} | {'Old Precision':<13} | {'New Precision':<13} | {'Old Recall':<10} | {'New Recall':<10} | {'Old mAP50':<9} | {'New mAP50':<9}")
    print("-" * 95)
    for name in ["RBC", "WBC", "Platelet"]:
        # Match class names properly (model might return capitalized or lowercase)
        key = name
        if key not in old_metrics:
            key = name.lower()
        if key not in old_metrics:
            # Fallback
            key = list(old_metrics.keys())[0]
            
        o_cls = old_metrics.get(key, {"precision": 0, "recall": 0, "map50": 0})
        n_cls = new_metrics.get(key, {"precision": 0, "recall": 0, "map50": 0})
        
        print(f"{name:<12} | {o_cls['precision']:<13.2%} | {n_cls['precision']:<13.2%} | {o_cls['recall']:<10.2%} | {n_cls['recall']:<10.2%} | {o_cls['map50']:<9.2%} | {n_cls['map50']:<9.2%}")
    print("=" * 95)
    
    # Save to metrics.json
    metrics_payload = {
        "old_model": {
            "per_class": old_metrics,
            "overall": old_overall
        },
        "new_model": {
            "per_class": new_metrics,
            "overall": new_overall
        },
        "comparison_note": "Evaluated on held-out test split of 56 images from BCCD (augmented training split)."
    }
    
    metrics_json_path = ROOT / "models" / "yolov8_blood" / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Metrics saved to {metrics_json_path}")
    
    # ── 5. Promote Improved Model to best.pt ──
    print("\n[STEP 5] Promoting improved model weights to production 'best.pt'...")
    shutil.copy2(str(improved_model_path), str(old_model_path))
    print(f"✓ Replaced {old_model_path} with {improved_model_path}")
    print("\nAll done!")

if __name__ == "__main__":
    run()
