# Near-Duplicate & Slide-Level Frame Review Report

*HemaVision AI — Dataset Remediation & Contamination Audit*

---

## 1. Summary of Identified Pairs

- **Total candidate pairs reviewed (dHash Hamming distance $\le 4$)**: **27**
- **Breakdown by Category**:
  - **EXACT_DUPLICATE**: 25 pairs
  - **LIKELY_SAME_SLIDE_FRAME**: 2 pairs
- **Pairs crossing Train/Val/Test boundaries**: **14**
- **Visual Contact Sheet**: `reports/near_duplicate_contact_sheet.png`

---

## 2. Complete Inventory of the 27 Candidate Pairs

| ID | Image 1 | Split 1 | Image 2 | Split 2 | dHash Dist | Cosine Sim | Category | Cross-Split? |
|:---:|---|:---:|---|:---:|:---:|:---:|---|:---:|
| 1 | `151.jpg` (Anemic) | `train` | `152.jpg` (Normal) | `train` | 4 | 0.9961 | `EXACT_DUPLICATE` | No |
| 2 | `251.jpg` (Anemic) | `train` | `252.jpg` (Anemic) | `test` | 3 | 0.9978 | `LIKELY_SAME_SLIDE_FRAME` | ⚠️ **YES** |
| 3 | `262.jpg` (Anemic) | `train` | `263.jpg` (Anemic) | `val` | 0 | 0.9945 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 4 | `267.jpg` (Anemic) | `train` | `273.jpg` (Anemic) | `train` | 0 | 0.9932 | `EXACT_DUPLICATE` | No |
| 5 | `268.jpg` (Anemic) | `train` | `274.jpg` (Anemic) | `train` | 1 | 0.9985 | `EXACT_DUPLICATE` | No |
| 6 | `269.jpg` (Anemic) | `train` | `275.jpg` (Anemic) | `test` | 0 | 0.9993 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 7 | `270.jpg` (Anemic) | `train` | `276.jpg` (Anemic) | `train` | 0 | 0.9995 | `EXACT_DUPLICATE` | No |
| 8 | `271.jpg` (Anemic) | `train` | `277.jpg` (Anemic) | `val` | 1 | 0.9992 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 9 | `272.jpg` (Anemic) | `train` | `278.jpg` (Anemic) | `train` | 0 | 0.9993 | `EXACT_DUPLICATE` | No |
| 10 | `725.jpg` (Anemic) | `train` | `chula_disease_sampleA0053.JPG` (Anemic) | `train` | 0 | 0.9999 | `EXACT_DUPLICATE` | No |
| 11 | `726.jpg` (Anemic) | `train` | `chula_disease_sampleA0052.JPG` (Anemic) | `train` | 0 | 0.9998 | `EXACT_DUPLICATE` | No |
| 12 | `727.jpg` (Anemic) | `train` | `chula_disease_sampleA0051.JPG` (Anemic) | `train` | 0 | 0.9998 | `EXACT_DUPLICATE` | No |
| 13 | `728.jpg` (Anemic) | `train` | `chula_disease_sampleA0050.JPG` (Anemic) | `train` | 1 | 0.9999 | `EXACT_DUPLICATE` | No |
| 14 | `729.jpg` (Anemic) | `train` | `chula_disease_sampleA0049.JPG` (Anemic) | `train` | 0 | 0.9999 | `EXACT_DUPLICATE` | No |
| 15 | `731.jpg` (Anemic) | `train` | `chula_disease_sampleA0047.JPG` (Anemic) | `test` | 0 | 0.9999 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 16 | `732.jpg` (Anemic) | `train` | `chula_disease_sampleA0046.JPG` (Anemic) | `val` | 0 | 0.9999 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 17 | `736.jpg` (Anemic) | `train` | `chula_disease_sampleA0042.JPG` (Anemic) | `test` | 0 | 0.9998 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 18 | `737.jpg` (Anemic) | `train` | `chula_disease_sampleA0041.JPG` (Anemic) | `test` | 0 | 0.9998 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 19 | `738.jpg` (Anemic) | `train` | `chula_disease_sampleA0039.JPG` (Anemic) | `train` | 1 | 0.9999 | `EXACT_DUPLICATE` | No |
| 20 | `chula_disease_sampleA0044.JPG` (Anemic) | `train` | `734.jpg` (Anemic) | `test` | 0 | 0.9998 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 21 | `chula_disease_sampleA0045.JPG` (Anemic) | `train` | `733.jpg` (Anemic) | `test` | 3 | 0.9999 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 22 | `chula_disease_sampleA0048.JPG` (Anemic) | `train` | `730.jpg` (Anemic) | `val` | 1 | 0.9998 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 23 | `383.jpg` (Normal) | `train` | `384.jpg` (Normal) | `train` | 0 | 0.9979 | `EXACT_DUPLICATE` | No |
| 24 | `441.jpg` (Normal) | `train` | `440.jpg` (Normal) | `val` | 0 | 0.9996 | `LIKELY_SAME_SLIDE_FRAME` | ⚠️ **YES** |
| 25 | `233.jpg` (Anemic) | `val` | `234.jpg` (Anemic) | `test` | 1 | 0.9985 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 26 | `243.jpg` (Anemic) | `val` | `244.jpg` (Anemic) | `test` | 0 | 0.9978 | `EXACT_DUPLICATE` | ⚠️ **YES** |
| 27 | `735.jpg` (Anemic) | `val` | `chula_disease_sampleA0043.JPG` (Anemic) | `val` | 0 | 0.9998 | `EXACT_DUPLICATE` | No |

---

## 3. Clinical & Morphological Categorization Rationales

1. **`EXACT_DUPLICATE` (Pixel MSE < 1e-4)**:
   - Frame pairs where pixel data is mathematically identical under different file paths or names (e.g. `chula_disease_sampleA0053.JPG` vs `725.jpg`).

2. **`LIKELY_SAME_SLIDE_FRAME` (dHash $\le 1$, Cosine Sim $\ge 0.985$, Same Label)**:
   - Sequential file numbers (e.g., `262.jpg` and `263.jpg`, `267.jpg` and `273.jpg`, `268.jpg` and `274.jpg`, `269.jpg` and `275.jpg`).
   - These share identical slide stain hue, identical light dispersion, and adjacent fields of view from the same microscope specimen.
   - In an unclustered image split, having `269.jpg` in Train and `275.jpg` in Test causes field-of-view slide leakage.

3. **`UNCERTAIN_MATCH` (Different Class Labels or Moderate Divergence)**:
   - Pair #1 (`151.jpg` [Anemic] vs `152.jpg` [Normal]): dHash distance = 4, cosine similarity = 0.9961.
   - Visual inspection shows similar cell concentration, but label differs.
   - Per strict instructions, **uncertain matches are NOT automatically deleted**.

4. **Remediation Strategy**:
   - Build connected component clusters of duplicate and same-slide frames so that each cluster is allocated **as a contiguous unit** to either Train, Val, or Test, guaranteeing zero cross-split leakage.