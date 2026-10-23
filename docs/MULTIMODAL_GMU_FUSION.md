# Gated Multimodal Fusion (GMU) Architecture for Anemia Screening

*HemaVision AI — Medical Deep Learning Multimodal Integration*

---

## 1. Architectural Motivation & Theoretical Foundation

In medical screening, peripheral blood smear photomicrographs and automated Complete Blood Count (CBC) numeric parameters provide complementary diagnostic perspectives:
- **Blood Smear Photomicrography**: Directly depicts erythrocyte cytomorphology, central pallor enlargement, anisocytosis, poikilocytosis, target cells, and sickle cells.
- **CBC Laboratory Parameters**: Quantifies physical erythrocyte indices (MCV, MCH, MCHC) and systemic hemoglobin levels.

To fuse these distinct data regimes without manual feature engineering, we implemented a **Gated Multimodal Unit (GMU)** architecture in PyTorch ([`src/models/gmu_fusion.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/src/models/gmu_fusion.py)).

---

## 2. Mathematical Formulation

```
Blood Smear Image (3 x 224 x 224)
        │
        ▼
MobileNetV3-Small Backbone
        │
        ▼
   Global Pool (576) ──► Linear Projection (128) ──► x_img (128-d)
                                                         │
                                                         ▼
                                                Candidate h_img = tanh(W_img * x_img + b_img)
                                                         │
                                                         │
Structured CBC Features (5-d)                            │
        │                                                ▼
  2-Layer MLP (64 -> 128) ──────────────► x_cbc ──► Candidate h_cbc = tanh(W_cbc * x_cbc + b_cbc)
                                                         │
                                                         ▼
                                       Dynamic Gate: z = σ(W_z * [h_img, h_cbc] + b_z)
                                                         │
                                                         ▼
                                  Fused Representation: h_fused = z ⊙ h_img + (1 - z) ⊙ h_cbc
                                                         │
                                                         ▼
                                               Classification Head (128 -> 64 -> 1)
                                                         │
                                                         ▼
                                                Anemia Probability P(Anemia)
```

### Mathematical Equations

1. **Candidate Modality Representations**:
   $$h_{\text{img}} = \tanh(W_{\text{img}} x_{\text{img}} + b_{\text{img}}), \quad h_{\text{img}} \in \mathbb{R}^{d}$$
   $$h_{\text{cbc}} = \tanh(W_{\text{cbc}} x_{\text{cbc}} + b_{\text{cbc}}), \quad h_{\text{cbc}} \in \mathbb{R}^{d}$$

2. **Gating Mechanism**:
   $$z = \sigma\left(W_z [h_{\text{img}}, h_{\text{cbc}}] + b_z\right) \in (0, 1)^{d}$$
   where $\sigma(u) = \frac{1}{1 + e^{-u}}$ is the sigmoid activation applied elementwise.

3. **Gated Multimodal Combination**:
   $$h_{\text{fused}} = z \odot h_{\text{img}} + (1 - z) \odot h_{\text{cbc}}$$

---

## 3. Explicit Missing-Modality Masking

In clinical reality, a patient may only have a blood smear slide (e.g., in a remote clinic with a tele-microscope) or only a laboratory CBC panel.

The GMU architecture implements explicit binary mask tensors:
$$(m_{\text{img}}, m_{\text{cbc}}) \in \{0, 1\}^2$$

- **Dual Modalities Present ($m_{\text{img}}=1, m_{\text{cbc}}=1$)**:
  The full GMU gating equation operates dynamically:
  $$h_{\text{fused}} = z \odot h_{\text{img}} + (1 - z) \odot h_{\text{cbc}}$$

- **Smear Only ($m_{\text{img}}=1, m_{\text{cbc}}=0$)**:
  The CBC candidate embedding is zeroed ($h_{\text{cbc}} = \mathbf{0}$) and the gate $z$ is forced to $\mathbf{1}$:
  $$h_{\text{fused}} = h_{\text{img}}$$

- **CBC Only ($m_{\text{img}}=0, m_{\text{cbc}}=1$)**:
  The image candidate embedding is zeroed ($h_{\text{img}} = \mathbf{0}$) and the gate $z$ is forced to $\mathbf{0}$:
  $$h_{\text{fused}} = h_{\text{cbc}}$$

- **Neither Present ($m_{\text{img}}=0, m_{\text{cbc}}=0$)**:
  $$h_{\text{fused}} = \mathbf{0}$$

This guarantees deterministic, crash-safe inference without numerical NaN propagation.

---

## 4. Medical Integrity Disclosures & Scope Boundaries

### A. Non-Negotiable Training Status
> [!IMPORTANT]
> **Multimodal Training Status**: `UNTRAINED_PENDING_PAIRED_DATA`
>
> An extensive repository audit verified that existing microscopy images (`data/anerbc_raw/`, `chula_main.zip`) and CBC records (`data/anemia/`) have **zero patient-level linkage**. 
> 
> Training a deep multimodal fusion model on arbitrarily or synthetically paired records creates illusory cross-modal correlations, invalidates empirical benchmarks, and is medically unsafe. Consequently, the GMU weights remain designated as **untrained** until a genuine, prospectively paired clinical cohort is acquired.

### B. Gate Weight Interpretation Guardrail
> [!WARNING]
> **Interpretation of Gate Activations**:
> GMU gate values $z \in (0, 1)^d$ represent intermediate neural routing activations across the latent feature space. They **must not** be displayed or interpreted as calibrated percentages of clinical diagnostic reliability or confidence.

---

## 5. Software Integration & Verification

The GMU implementation is verified with automated tests in [`tests/test_gmu_architecture.py`](file:///c:/Users/HP/Desktop/Blood_report_analyzer-main/tests/test_gmu_architecture.py):
- Forward pass tensor dimension consistency (`B x 128`)
- Pre-extracted feature vector vs raw image input compatibility
- Mask enforcement for image-only and CBC-only operational modes
- Gradient propagation from binary cross-entropy loss through classification head and gating layers
- Serialization to `models/anemia_fusion/gmu_fusion_untrained.pth` and `models/anemia_fusion/gmu_metadata.json`
