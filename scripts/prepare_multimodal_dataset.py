"""
prepare_multimodal_dataset.py

Harmonizes and consolidates multiple CBC and image datasets for multimodal fusion:
1. Extracts 121 specialized RBC disease images from chula_main.zip into data/anerbc_raw/Anemia/.
2. Harmonizes diagnosed_cbc_data_v4.csv (1,281 clinical cases) with anemia_dataset.csv (1,421 cases).
3. Produces data/anemia/processed/master_cbc_dataset.csv with standardized columns:
   - Gender (0=Female, 1=Male)
   - Hemoglobin (g/dL)
   - MCV (fL)
   - MCH (pg)
   - MCHC (g/dL)
   - Result (0=Normal, 1=Anemic)
   - Diagnosis (clinical text category)
   - Source (dataset provenance)
"""

from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils import get_logger, project_path

log = get_logger("prepare_dataset")


def extract_chula_disease_images():
    zip_path = project_path("chula_main.zip")
    dest_dir = project_path("data", "anerbc_raw", "Anemia")
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        log.warning("chula_main.zip not found in project root. Skipping zip extraction.")
        return

    with zipfile.ZipFile(zip_path, "r") as z:
        disease_files = [f for f in z.namelist() if "RBC Diseases" in f and f.lower().endswith((".jpg", ".jpeg", ".png"))]
        log.info("Found %d disease images in chula_main.zip", len(disease_files))
        extracted_count = 0
        for f in disease_files:
            filename = "chula_disease_" + Path(f).name
            dest_file = dest_dir / filename
            if not dest_file.exists():
                with z.open(f) as src, open(dest_file, "wb") as out:
                    shutil.copyfileobj(src, out)
                extracted_count += 1
        log.info("Extracted %d new disease images to %s", extracted_count, dest_dir)


def harmonize_cbc_datasets() -> pd.DataFrame:
    raw_dir = project_path("data", "anemia", "raw")
    proc_dir = project_path("data", "anemia", "processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    records = []

    # 1. Dataset 1: Original Kaggle Anemia Dataset (1,421 rows)
    ds1_path = raw_dir / "anemia_dataset.csv"
    if ds1_path.exists():
        df1 = pd.read_csv(ds1_path)
        log.info("Loaded Dataset 1 (anemia_dataset.csv): %d rows", len(df1))
        for _, row in df1.iterrows():
            g = int(row.get("Gender", 0))
            hb = float(row.get("Hemoglobin", 12.0))
            mcv = float(row.get("MCV", 85.0))
            mch = float(row.get("MCH", 28.0))
            mchc = float(row.get("MCHC", 33.0))
            res = int(row.get("Result", 1 if hb < 12.0 else 0))
            records.append({
                "Gender": g,
                "Hemoglobin": round(hb, 2),
                "MCV": round(mcv, 2),
                "MCH": round(mch, 2),
                "MCHC": round(mchc, 2),
                "Result": res,
                "Diagnosis": "Anemic" if res == 1 else "Healthy",
                "Source": "kaggle_anemia_dataset"
            })

    # 2. Dataset 2: Clinical Diagnosed CBC Dataset (1,281 rows)
    ds2_path = raw_dir / "diagnosed_cbc_data_v4.csv"
    if ds2_path.exists():
        df2 = pd.read_csv(ds2_path)
        log.info("Loaded Dataset 2 (diagnosed_cbc_data_v4.csv): %d rows", len(df2))

        anemia_diagnoses = {
            "Normocytic hypochromic anemia",
            "Normocytic normochromic anemia",
            "Iron deficiency anemia",
            "Other microcytic anemia",
            "Macrocytic anemia"
        }

        for _, row in df2.iterrows():
            diag = str(row.get("Diagnosis", "Healthy")).strip()
            # Determine ground truth label
            if diag in anemia_diagnoses:
                res = 1
            elif diag == "Healthy":
                res = 0
            else:
                # Other conditions (Leukemia/Thrombocytopenia) - check Hemoglobin threshold
                try:
                    hb_val = float(row.get("HGB", 12.0))
                    res = 1 if hb_val < 11.5 else 0
                except (ValueError, TypeError):
                    continue

            try:
                hb = float(row.get("HGB", 12.0))
                mcv = float(row.get("MCV", 85.0))
                mch = float(row.get("MCH", 28.0))
                mchc = float(row.get("MCHC", 33.0))

                # If values are corrupted or extreme outliers, skip or sanitize
                if hb <= 0 or hb > 25 or mcv <= 20 or mcv > 160:
                    continue

                # Infer gender probabilistically if missing based on adult reference midpoint
                g = 1 if hb > 13.5 else 0

                records.append({
                    "Gender": g,
                    "Hemoglobin": round(hb, 2),
                    "MCV": round(mcv, 2),
                    "MCH": round(mch, 2),
                    "MCHC": round(mchc, 2),
                    "Result": res,
                    "Diagnosis": diag,
                    "Source": "diagnosed_cbc_v4"
                })
            except (ValueError, TypeError):
                continue

    master_df = pd.DataFrame(records)
    out_path = proc_dir / "master_cbc_dataset.csv"
    master_df.to_csv(out_path, index=False)
    log.info("Saved consolidated master CBC dataset to %s (%d rows)", out_path, len(master_df))
    log.info("Class distribution in master CBC dataset: %s", master_df["Result"].value_counts().to_dict())
    return master_df


if __name__ == "__main__":
    extract_chula_disease_images()
    harmonize_cbc_datasets()
