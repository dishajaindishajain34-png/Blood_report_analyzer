# Data Leakage Check — AneRBC Dataset

*Generated automatically by `scripts/inspect_anerbc_dataset.py`*

---

## 1. Dataset Root

```
data\anerbc_raw
```

## 2. Image Counts

| Class | Count |
|-------|-------|
| Anemia | 613 |
| Normal | 125 |
| **Total** | **738** |

## 3. Duplicate Check

- Duplicate groups found: **22**
- ⚠️ Duplicates detected — removed from training set.

## 4. Sample/Patient Grouping

- Sample groups detected from filenames: **0**
- Groups with multiple images: **0**

✅ No multi-image groups detected. Image-level stratified split is acceptable.

## 5. Train/Val/Test Split Strategy

- **Stratified image-level split** (StratifiedShuffleSplit by class label)
- Split ratio: 70% train / 15% val / 15% test
- Random seed: fixed (42) for reproducibility
- **Test set is held out until all model decisions are finalized**

## 6. CBC Metadata Pairing

- CBC/metadata CSV found: **No**
- Image–CBC pairing feasible: **No — image model only until paired data is verified**

## 7. Preprocessing Leakage Prevention

- Normalization statistics (mean/std) are computed on **training split only**
- Applied to val/test splits using training stats — no leakage
- Augmentations applied only to training split

## 8. Test Set Isolation

- Test set is NOT used for:
  - Hyperparameter tuning
  - Model selection
  - Threshold selection
  - Feature selection
- Test set is used **only** for final evaluation after all model decisions are frozen

## 9. Image–CBC Pairing Verification

- Pairing NOT verified — no matching CBC CSV found. Fusion model will only be trained when valid paired data is available. Do NOT fabricate or randomly match image–CBC pairs.
