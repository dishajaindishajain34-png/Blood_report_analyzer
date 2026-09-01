"""
Test script to run detection on sample blood images using the trained model.
"""

import sys
from pathlib import Path
from PIL import Image
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.detect import detect_cells

def test_detection():
    """Run detection on all sample images."""
    sample_dir = project_root / "data" / "sample_images"
    results_summary = []
    
    print("=" * 80)
    print("BLOOD CELL DETECTION - SAMPLE IMAGES TEST")
    print("=" * 80)
    print(f"\nModel: models/yolov8_blood/best.pt")
    print(f"Test Images Directory: {sample_dir}\n")
    
    for img_path in sorted(sample_dir.glob("*.jpg")):
        print(f"\n{'─' * 80}")
        print(f"Processing: {img_path.name}")
        print(f"{'─' * 80}")
        
        try:
            # Load image
            img = Image.open(img_path)
            print(f"Image size: {img.size}")
            
            # Run detection
            result = detect_cells(img, conf_threshold=0.35, iou_threshold=0.40)
            
            # Display results
            print(f"\n✓ Detection successful!")
            print(f"  Total cells detected: {result.total_cells}")
            print(f"  Breakdown:")
            print(f"    - RBC:      {result.raw_counts['rbc']}")
            print(f"    - WBC:      {result.raw_counts['wbc']}")
            print(f"    - Platelet: {result.raw_counts['platelet']}")
            print(f"  Average confidence: {result.avg_confidence:.2%}")
            print(f"  Model: {result.model_name}")
            
            # Save annotated image
            output_path = project_root / "reports" / f"detection_{img_path.stem}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.annotated_pil.save(output_path)
            print(f"\n  ✓ Annotated image saved: {output_path.relative_to(project_root)}")
            
            # Store for summary
            results_summary.append({
                "image": img_path.name,
                "total_cells": result.total_cells,
                "counts": result.raw_counts,
                "avg_confidence": result.avg_confidence,
                "detections_count": len(result.detections)
            })
            
        except Exception as e:
            print(f"\n✗ Error processing {img_path.name}:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary report
    print(f"\n\n{'=' * 80}")
    print("DETECTION SUMMARY REPORT")
    print(f"{'=' * 80}\n")
    
    if results_summary:
        total_images = len(results_summary)
        total_cells_all = sum(r["total_cells"] for r in results_summary)
        avg_cells = total_cells_all / total_images if total_images > 0 else 0
        
        print(f"Images processed: {total_images}")
        print(f"Total cells detected across all images: {total_cells_all}")
        print(f"Average cells per image: {avg_cells:.1f}\n")
        
        print("Breakdown by cell type (across all images):")
        total_rbc = sum(r["counts"]["rbc"] for r in results_summary)
        total_wbc = sum(r["counts"]["wbc"] for r in results_summary)
        total_platelet = sum(r["counts"]["platelet"] for r in results_summary)
        
        print(f"  - RBC:      {total_rbc} ({total_rbc/total_cells_all*100:.1f}%)")
        print(f"  - WBC:      {total_wbc} ({total_wbc/total_cells_all*100:.1f}%)")
        print(f"  - Platelet: {total_platelet} ({total_platelet/total_cells_all*100:.1f}%)")
        
        print(f"\nDetailed Results:")
        print(f"{'Image':<25} {'Total':<8} {'RBC':<6} {'WBC':<6} {'Platelet':<10} {'Avg Conf':<10}")
        print("─" * 70)
        for r in results_summary:
            print(f"{r['image']:<25} {r['total_cells']:<8} {r['counts']['rbc']:<6} {r['counts']['wbc']:<6} {r['counts']['platelet']:<10} {r['avg_confidence']:<10.2%}")
        
        # Save detailed results to JSON
        json_output = project_root / "reports" / "detection_results.json"
        json_output.parent.mkdir(parents=True, exist_ok=True)
        with open(json_output, "w") as f:
            json.dump(results_summary, f, indent=2)
        print(f"\n✓ Detailed results saved: {json_output.relative_to(project_root)}")
    else:
        print("No images were successfully processed.")

if __name__ == "__main__":
    test_detection()
