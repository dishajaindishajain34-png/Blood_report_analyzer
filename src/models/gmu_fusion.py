"""
gmu_fusion.py — Gated Multimodal Fusion (GMU) Architecture for Anemia Screening.

Implements an advanced deep multimodal architecture combining blood-smear microscopy
features (MobileNetV3-Small) and structured complete blood count (CBC) laboratory features
via a Gated Multimodal Unit (GMU) with explicit missing-modality mask enforcement.

Medical & Integrity Constraints:
1. This model architecture is defined for prospective paired clinical cohorts.
2. In the absence of genuine patient-level paired data, this model is explicitly
   designated as UNTRAINED_PENDING_PAIRED_DATA.
3. Gate weights represent latent layer routing coefficients, NOT calibrated percentages
   of modality diagnostic reliability.
4. Inference is strictly rejected when both modalities are missing.
5. Missing branches are zero-masked and prevented from influencing predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from src.utils import get_logger, project_path

log = get_logger(__name__)


class ImageBranch(nn.Module):
    """
    Image feature extraction branch.
    Accepts either raw 3-channel images (B, 3, 224, 224) or pre-extracted embeddings (B, 576).
    Projects to latent dimension d_latent (default: 128).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        pretrained_weights_path: Optional[Path] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Backbone: MobileNetV3-Small
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        if pretrained_weights_path and Path(pretrained_weights_path).exists():
            try:
                state_dict = torch.load(pretrained_weights_path, map_location="cpu", weights_only=True)
                filtered_state = {k: v for k, v in state_dict.items() if not k.startswith("classifier.3")}
                backbone.load_state_dict(filtered_state, strict=False)
                log.info("Loaded pretrained MobileNetV3 backbone weights into ImageBranch.")
            except Exception as e:
                log.warning("Could not load fine-tuned image weights: %s. Using ImageNet weights.", e)

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        backbone_dim = 576  # MobileNetV3-Small pooled output dimension

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Dropout(p=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 3, H, W) or pre-extracted pooled features (B, 576).
        Returns:
            Tensor of shape (B, latent_dim).
        """
        if x.dim() == 4:
            feat = self.features(x)
            pooled = self.avgpool(feat)
            flattened = torch.flatten(pooled, 1)
        elif x.dim() == 2:
            flattened = x
        else:
            raise ValueError(f"Expected 2D or 4D tensor for image branch, got {x.dim()}D (shape {x.shape})")

        return self.projection(flattened)


class CBCBranch(nn.Module):
    """
    Structured laboratory CBC parameter extraction branch.
    Accepts (B, cbc_dim) features (e.g., Gender, Hemoglobin, MCV, MCH, MCHC).
    Projects to latent dimension d_latent (default: 128).
    """

    def __init__(self, in_features: int = 5, latent_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.15),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Dropout(p=0.15),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, in_features).
        Returns:
            Tensor of shape (B, latent_dim).
        """
        return self.net(x)


class GatedMultimodalUnit(nn.Module):
    """
    Gated Multimodal Unit (GMU) with explicit missing-modality mask enforcement.

    Transformations:
      h_img = tanh(W_img * x_img + b_img)
      h_cbc = tanh(W_cbc * x_cbc + b_cbc)

    Dynamic Gating:
      z = sigmoid(W_z * [h_img, h_cbc] + b_z)

    Fusion:
      h_fused = z * h_img + (1 - z) * h_cbc

    Missing-Modality Masking (m_img, m_cbc in {0, 1}):
      - If m_img == 0: image embedding zeroed, gate z forced to 0 -> h_fused = h_cbc
      - If m_cbc == 0: CBC embedding zeroed, gate z forced to 1 -> h_fused = h_img
      - If both missing: inference rejected with ValueError.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # Modality-specific candidate representations
        self.fc_img = nn.Linear(latent_dim, latent_dim)
        self.fc_cbc = nn.Linear(latent_dim, latent_dim)

        # Dynamic gating network
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x_img: torch.Tensor,
        x_cbc: torch.Tensor,
        mask_img: Optional[torch.Tensor] = None,
        mask_cbc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x_img.size(0)
        device = x_img.device

        if mask_img is None:
            mask_img = torch.ones((batch_size, 1), device=device, dtype=x_img.dtype)
        if mask_cbc is None:
            mask_cbc = torch.ones((batch_size, 1), device=device, dtype=x_cbc.dtype)

        # Check for invalid cases where both modalities are missing
        both_missing = (mask_img == 0) & (mask_cbc == 0)
        if both_missing.any():
            raise ValueError(
                "Inference rejected: both image and CBC modalities are missing for one or more samples. "
                "At least one valid modality is required."
            )

        # Expand masks across feature dimension
        m_img = mask_img.expand(-1, self.latent_dim)
        m_cbc = mask_cbc.expand(-1, self.latent_dim)

        # Apply candidate transformations with strict masking (missing branches are zeroed out)
        h_img = torch.tanh(self.fc_img(x_img)) * m_img
        h_cbc = torch.tanh(self.fc_cbc(x_cbc)) * m_cbc

        # Compute dynamic gate activation
        concat_h = torch.cat([h_img, h_cbc], dim=-1)
        z_raw = self.gate(concat_h)

        # Enforce missing-modality routing
        both_present = m_img * m_cbc
        img_only = m_img * (1.0 - m_cbc)
        cbc_only = (1.0 - m_img) * m_cbc

        # Forced gate values:
        # If image-only (m_img=1, m_cbc=0): z is forced to 1.0 (100% image routing)
        # If CBC-only   (m_img=0, m_cbc=1): z is forced to 0.0 (100% CBC routing)
        z_enforced = (z_raw * both_present) + (1.0 * img_only) + (0.0 * cbc_only)

        # Weighted combination: strictly ensures zero contribution from missing modality
        h_fused = (z_enforced * h_img) + ((1.0 - z_enforced) * h_cbc)

        return h_fused, z_enforced


class GMUAnemiaClassifier(nn.Module):
    """
    End-to-End Multimodal Anemia Classifier with GMU Fusion.
    """

    def __init__(
        self,
        cbc_features: int = 5,
        latent_dim: int = 128,
        image_pretrained_path: Optional[Path] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cbc_features = cbc_features

        self.image_branch = ImageBranch(latent_dim=latent_dim, pretrained_weights_path=image_pretrained_path)
        self.cbc_branch = CBCBranch(in_features=cbc_features, latent_dim=latent_dim)
        self.gmu = GatedMultimodalUnit(latent_dim=latent_dim)

        # Classification Head: latent_dim -> 64 -> 1
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        img: torch.Tensor,
        cbc: torch.Tensor,
        mask_img: Optional[torch.Tensor] = None,
        mask_cbc: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = img.size(0)
        device = img.device

        if mask_img is None:
            mask_img = torch.ones((batch_size, 1), device=device, dtype=img.dtype)
        if mask_cbc is None:
            mask_cbc = torch.ones((batch_size, 1), device=device, dtype=cbc.dtype)

        both_missing = (mask_img == 0) & (mask_cbc == 0)
        if both_missing.any():
            raise ValueError(
                "Inference rejected: both image and CBC modalities are missing for one or more samples. "
                "Multimodal GMU requires at least one present modality."
            )

        feat_img = self.image_branch(img)
        feat_cbc = self.cbc_branch(cbc)

        fused_embedding, gate_z = self.gmu(
            feat_img, feat_cbc, mask_img=mask_img, mask_cbc=mask_cbc
        )

        logits = self.classifier(fused_embedding)
        probs = torch.sigmoid(logits)

        return {
            "logits": logits,
            "probability": probs,
            "latent_gate_routing": gate_z,
            "latent_gate_mean": gate_z.mean(dim=-1, keepdim=True),
            "fused_embedding": fused_embedding,
        }


def save_untrained_gmu_checkpoint():
    """
    Instantiate GMU architecture and serialize checkpoint and metadata
    with explicit designation as UNTRAINED_PENDING_PAIRED_DATA.
    """
    model_dir = project_path("models", "anemia_fusion")
    model_dir.mkdir(parents=True, exist_ok=True)

    image_weights = project_path("models", "anemia_image", "mobilenetv3_anemia.pth")
    model = GMUAnemiaClassifier(
        cbc_features=5,
        latent_dim=128,
        image_pretrained_path=image_weights if image_weights.exists() else None,
    )

    checkpoint_path = model_dir / "gmu_fusion_untrained.pth"
    torch.save(model.state_dict(), checkpoint_path)

    metadata = {
        "model_name": "Gated Multimodal Unit (GMU) Anemia Architecture",
        "model_version": "1.0.0-architecture-only",
        "training_status": "UNTRAINED_PENDING_PAIRED_DATA",
        "latent_dimension": 128,
        "image_backbone": "MobileNetV3-Small",
        "cbc_input_features": ["Gender", "Hemoglobin", "MCV", "MCH", "MCHC"],
        "missing_modality_masking_supported": True,
        "inference_rejection_policy": "Inference is strictly rejected if both image and CBC modalities are missing.",
        "clinical_disclaimer": (
            "This PyTorch GMU model is an architectural implementation with verified forward-pass "
            "and missing-modality tensor masking. It has NOT been trained on arbitrarily matched data. "
            "Training and clinical evaluation remain strictly blocked until verified patient-level "
            "paired smear-CBC clinical cohorts are acquired."
        ),
        "gate_interpretation_notice": (
            "GMU gate activations represent intermediate neural network routing weights across the latent "
            "embedding space. They MUST NOT be interpreted as calibrated clinical percentages of modality "
            "diagnostic reliability."
        ),
    }

    meta_path = model_dir / "gmu_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log.info("Saved GMU architecture weights -> %s", checkpoint_path)
    log.info("Saved GMU metadata -> %s", meta_path)
    return True


if __name__ == "__main__":
    save_untrained_gmu_checkpoint()
