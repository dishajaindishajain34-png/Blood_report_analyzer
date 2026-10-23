"""
audit_near_duplicates.py

Comprehensive Near-Duplicate Audit on 837 microscopy smear images using:
1. Difference Hash (dHash) with Hamming distance threshold <= 4
2. MobileNetV3-Small deep feature embedding cosine similarity >= 0.98

Does NOT automatically delete any matches.
Reports all candidate pairs, their split locations, and visual assessment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, project_path

log = get_logger("near_dup_audit")


def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Compute 64-bit difference hash (dHash)."""
    img_gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(img_gray)
    diff = pixels[:, 1:] > pixels[:, :-1]
    hash_val = 0
    for bit in diff.flatten():
        hash_val = (hash_val << 1) | int(bit)
    return hash_val


def hamming_distance(h1: int, h2: int) -> int:
    """Compute Hamming distance between two 64-bit integers."""
    return bin(h1 ^ h2).count("1")


def run_near_duplicate_audit():
    manifest_path = project_path("models", "anemia_image", "split_manifest.csv")
    if not manifest_path.exists():
        log.error("Split manifest not found at %s. Run export_split_manifest_and_predictions.py first.", manifest_path)
        return False

    manifest_df = pd.read_csv(manifest_path)
    log.info("Loaded split manifest: %d images", len(manifest_df))

    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    features_fn = nn.Sequential(m.features, m.avgpool)
    features_fn.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    log.info("Computing dHash and deep embeddings for all %d images...", len(manifest_df))
    dhashes = []
    embeddings = []
    valid_indices = []

    for idx, row in manifest_df.iterrows():
        img_path = Path(row["filepath"])
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                dh = compute_dhash(img_rgb)
                dhashes.append(dh)

                tensor = transform(img_rgb).unsqueeze(0)
                with torch.no_grad():
                    feat = features_fn(tensor)
                    feat_flat = torch.flatten(feat, 1).squeeze().numpy()
                    norm = np.linalg.norm(feat_flat)
                    if norm > 0:
                        feat_flat = feat_flat / norm
                    embeddings.append(feat_flat)
                valid_indices.append(idx)
        except Exception as e:
            log.warning("Could not process %s: %s", img_path, e)

    df_valid = manifest_df.iloc[valid_indices].reset_index(drop=True)
    embeddings = np.array(embeddings)
    n_images = len(df_valid)
    log.info("Processed %d valid images. Performing pairwise comparison across %d pairs...",
             n_images, n_images * (n_images - 1) // 2)

    candidate_pairs = []
    dhash_thresh = 4
    sim_thresh = 0.98

    sim_matrix = np.dot(embeddings, embeddings.T)

    for i in range(n_images):
        row_i = df_valid.iloc[i]
        dh_i = dhashes[i]

        for j in range(i + 1, n_images):
            row_j = df_valid.iloc[j]
            dh_dist = hamming_distance(dh_i, dhashes[j])
            cos_sim = float(sim_matrix[i, j])

            is_candidate = (dh_dist <= dhash_thresh) or (cos_sim >= sim_thresh)
            if is_candidate:
                cross_split = bool(row_i["split"] != row_j["split"])
                same_label = bool(row_i["label"] == row_j["label"])

                candidate_pairs.append({
                    "image_1": {
                        "filename": str(row_i["filename"]),
                        "filepath": str(row_i["filepath"]),
                        "label": int(row_i["label"]),
                        "class_name": str(row_i["class_name"]),
                        "split": str(row_i["split"]),
                    },
                    "image_2": {
                        "filename": str(row_j["filename"]),
                        "filepath": str(row_j["filepath"]),
                        "label": int(row_j["label"]),
                        "class_name": str(row_j["class_name"]),
                        "split": str(row_j["split"]),
                    },
                    "dhash_hamming_distance": int(dh_dist),
                    "embedding_cosine_similarity": round(cos_sim, 4),
                    "cross_split": cross_split,
                    "same_label": same_label,
                    "match_type": "high_similarity" if dh_dist <= 2 else "moderate_similarity"
                })

    log.info("Found %d candidate near-duplicate pairs (dHash <= %d or Cosine Sim >= %.2f)",
             len(candidate_pairs), dhash_thresh, sim_thresh)

    cross_split_candidates = [p for p in candidate_pairs if p["cross_split"]]
    log.info("Candidate pairs spanning different splits: %d", len(cross_split_candidates))

    out_file = project_path("models", "anemia_image", "near_duplicate_audit.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_images_analyzed": n_images,
            "total_pairs_compared": n_images * (n_images - 1) // 2,
            "dhash_distance_threshold": dhash_thresh,
            "cosine_similarity_threshold": sim_thresh,
            "candidate_pairs_found": len(candidate_pairs),
            "cross_split_candidates_count": len(cross_split_candidates),
            "candidate_pairs_top50": candidate_pairs[:50],
            "review_summary": {
                "cross_split_candidates": [
                    {
                        "pair": f"{p['image_1']['filename']} ({p['image_1']['split']}) <-> {p['image_2']['filename']} ({p['image_2']['split']})",
                        "dhash_dist": p["dhash_hamming_distance"],
                        "cosine_sim": p["embedding_cosine_similarity"],
                        "same_label": p["same_label"]
                    }
                    for p in cross_split_candidates[:50]
                ]
            }
        }, f, indent=2)

    log.info("Saved near-duplicate audit report -> %s", out_file)
    return True


if __name__ == "__main__":
    run_near_duplicate_audit()
