"""
create_cluster_level_split.py

Creates an image-group-level (cluster-level) leakage-aware split:
1. Loads 837 images from split_manifest.csv and near-duplicate pairs from near_duplicate_review.json.
2. Constructs connected components (clusters) of confirmed duplicates and likely same-slide frames.
3. Groups all related frames into atomic clusters so that no frames from the same cluster can cross splits.
4. Performs stratified cluster-level split (70% Train / 15% Val / 15% Test) with fixed seed=42.
5. Saves models/anemia_image_cluster_split/cluster_split_manifest.csv.
6. Verifies zero cross-split cluster leakage.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, project_path

log = get_logger("cluster_split")


def build_clusters(manifest_df: pd.DataFrame, review_json_path: Path) -> dict[str, str]:
    """
    Union-Find connected component clustering for same-slide and duplicate frames.
    """
    parent = {row["filename"]: row["filename"] for _, row in manifest_df.iterrows()}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # Load near-duplicate review pairs
    if review_json_path.exists():
        with open(review_json_path, "r", encoding="utf-8") as f:
            review_data = json.load(f)

        for p in review_data.get("pairs", []):
            # Only merge pairs of the SAME class label that are duplicates or same-slide
            if p.get("same_label", False) and p.get("category") in ["EXACT_DUPLICATE", "LIKELY_SAME_SLIDE_FRAME"]:
                f1 = p["img1_filename"]
                f2 = p["img2_filename"]
                if f1 in parent and f2 in parent:
                    union(f1, f2)
                    log.info("Merged cluster: %s <-> %s (dHash: %d, cos_sim: %.4f)",
                             f1, f2, p["dhash_distance"], p["cosine_similarity"])

    # Map each root to a cluster ID
    clusters = defaultdict(list)
    for fn in parent:
        clusters[find(fn)].append(fn)

    file_to_cluster = {}
    for c_idx, (root_fn, members) in enumerate(sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True), 1):
        c_id = f"cluster_{c_idx:04d}"
        for m in members:
            file_to_cluster[m] = c_id

    return file_to_cluster


def create_remediated_split(seed: int = 42):
    manifest_in = project_path("models", "anemia_image", "split_manifest.csv")
    review_in = project_path("reports", "near_duplicate_review.json")
    out_dir = project_path("models", "anemia_image_cluster_split")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_in)
    log.info("Loaded base manifest: %d images", len(df))

    # Assign clusters
    file_to_cluster = build_clusters(df, review_in)
    df["cluster_id"] = df["filename"].map(file_to_cluster)

    cluster_sizes = df["cluster_id"].value_counts().to_dict()
    df["cluster_size"] = df["cluster_id"].map(cluster_sizes)

    multi_frame_clusters = {c: s for c, s in cluster_sizes.items() if s > 1}
    log.info("Total image clusters: %d (Singletons: %d, Multi-frame clusters: %d)",
             len(cluster_sizes), len(cluster_sizes) - len(multi_frame_clusters), len(multi_frame_clusters))

    # Stratified cluster-level partitioning (70% train / 15% val / 15% test)
    rng = random.Random(seed)
    cluster_labels = df.groupby("cluster_id")["label"].agg(lambda s: s.mode()[0]).to_dict()

    train_clusters, val_clusters, test_clusters = set(), set(), set()

    for lbl in sorted(df["label"].unique()):
        clusters_for_lbl = [c for c, l in cluster_labels.items() if l == lbl]
        rng.shuffle(clusters_for_lbl)

        n_c = len(clusters_for_lbl)
        n_train = int(0.70 * n_c)
        n_val = int(0.15 * n_c)

        train_clusters.update(clusters_for_lbl[:n_train])
        val_clusters.update(clusters_for_lbl[n_train:n_train + n_val])
        test_clusters.update(clusters_for_lbl[n_train + n_val:])

    # Assign split to dataframe
    split_map = {}
    for c in train_clusters:
        split_map[c] = "train"
    for c in val_clusters:
        split_map[c] = "val"
    for c in test_clusters:
        split_map[c] = "test"

    df["cluster_split"] = df["cluster_id"].map(split_map)

    # Verification of zero leakage
    train_c = set(df[df["cluster_split"] == "train"]["cluster_id"])
    val_c = set(df[df["cluster_split"] == "val"]["cluster_id"])
    test_c = set(df[df["cluster_split"] == "test"]["cluster_id"])

    assert len(train_c.intersection(val_c)) == 0, "Leakage between Train and Val clusters!"
    assert len(train_c.intersection(test_c)) == 0, "Leakage between Train and Test clusters!"
    assert len(val_c.intersection(test_c)) == 0, "Leakage between Val and Test clusters!"

    log.info("\n--- REMEDIATED CLUSTER-LEVEL SPLIT SUMMARY ---")
    for s in ["train", "val", "test"]:
        sub = df[df["cluster_split"] == s]
        norm_count = int((sub["label"] == 0).sum())
        anem_count = int((sub["label"] == 1).sum())
        c_count = sub["cluster_id"].nunique()
        log.info("  %-6s: %3d images (%3d Normal, %3d Anemic) across %3d clusters",
                 s.upper(), len(sub), norm_count, anem_count, c_count)

    out_csv = out_dir / "cluster_split_manifest.csv"
    cols = ["filepath", "filename", "label", "class_name", "md5", "cluster_id", "cluster_size", "cluster_split"]
    df[cols].to_csv(out_csv, index=False)
    log.info("Saved remediated cluster split manifest -> %s", out_csv)

    return True


if __name__ == "__main__":
    create_remediated_split(seed=42)
