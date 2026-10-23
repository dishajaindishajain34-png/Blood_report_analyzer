"""
test_gmu_architecture.py

Unit tests for Gated Multimodal Fusion (GMU) architecture, tensor dimensions,
gradient propagation, explicit missing-modality mask enforcement, and rejection
when both modalities are missing.
"""

from __future__ import annotations

import unittest
import torch
import torch.nn as nn

from src.models.gmu_fusion import (
    CBCBranch,
    GatedMultimodalUnit,
    GMUAnemiaClassifier,
    ImageBranch,
)


class TestGMUArchitecture(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.latent_dim = 128
        self.cbc_features = 5

        # Dummy inputs
        self.raw_images = torch.randn(self.batch_size, 3, 224, 224)
        self.precomputed_features = torch.randn(self.batch_size, 576)
        self.cbc_data = torch.randn(self.batch_size, self.cbc_features)

    def test_image_branch_forward_raw(self):
        branch = ImageBranch(latent_dim=self.latent_dim, freeze_backbone=True)
        out = branch(self.raw_images)
        self.assertEqual(out.shape, (self.batch_size, self.latent_dim))
        self.assertFalse(torch.isnan(out).any())

    def test_image_branch_forward_precomputed(self):
        branch = ImageBranch(latent_dim=self.latent_dim, freeze_backbone=True)
        out = branch(self.precomputed_features)
        self.assertEqual(out.shape, (self.batch_size, self.latent_dim))
        self.assertFalse(torch.isnan(out).any())

    def test_cbc_branch_forward(self):
        branch = CBCBranch(in_features=self.cbc_features, latent_dim=self.latent_dim)
        out = branch(self.cbc_data)
        self.assertEqual(out.shape, (self.batch_size, self.latent_dim))
        self.assertFalse(torch.isnan(out).any())

    def test_gmu_dual_modality(self):
        gmu = GatedMultimodalUnit(latent_dim=self.latent_dim)
        x_img = torch.randn(self.batch_size, self.latent_dim)
        x_cbc = torch.randn(self.batch_size, self.latent_dim)

        fused, gate_z = gmu(x_img, x_cbc)
        self.assertEqual(fused.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(gate_z.shape, (self.batch_size, self.latent_dim))
        self.assertTrue((gate_z >= 0.0).all() and (gate_z <= 1.0).all())
        self.assertFalse(torch.isnan(fused).any())

    def test_gmu_missing_image_mask(self):
        """When image is missing (mask_img=0, mask_cbc=1), gate is forced to 0 (100% CBC)."""
        gmu = GatedMultimodalUnit(latent_dim=self.latent_dim)
        x_img = torch.randn(self.batch_size, self.latent_dim)
        x_cbc = torch.randn(self.batch_size, self.latent_dim)

        mask_img = torch.zeros(self.batch_size, 1)
        mask_cbc = torch.ones(self.batch_size, 1)

        fused, gate_z = gmu(x_img, x_cbc, mask_img=mask_img, mask_cbc=mask_cbc)

        # Gate z should be forced to 0 (all routing to CBC)
        self.assertTrue(torch.allclose(gate_z, torch.zeros_like(gate_z)))

        # Fused must strictly match candidate CBC representation
        expected_cbc = torch.tanh(gmu.fc_cbc(x_cbc))
        self.assertTrue(torch.allclose(fused, expected_cbc, atol=1e-5))

    def test_gmu_missing_cbc_mask(self):
        """When CBC is missing (mask_img=1, mask_cbc=0), gate is forced to 1 (100% Image)."""
        gmu = GatedMultimodalUnit(latent_dim=self.latent_dim)
        x_img = torch.randn(self.batch_size, self.latent_dim)
        x_cbc = torch.randn(self.batch_size, self.latent_dim)

        mask_img = torch.ones(self.batch_size, 1)
        mask_cbc = torch.zeros(self.batch_size, 1)

        fused, gate_z = gmu(x_img, x_cbc, mask_img=mask_img, mask_cbc=mask_cbc)

        # Gate z should be forced to 1 (all routing to Image)
        self.assertTrue(torch.allclose(gate_z, torch.ones_like(gate_z)))

        # Fused must strictly match candidate Image representation
        expected_img = torch.tanh(gmu.fc_img(x_img))
        self.assertTrue(torch.allclose(fused, expected_img, atol=1e-5))

    def test_missing_branch_cannot_influence_predictions(self):
        """
        Verify safeguard: an arbitrary change to a missing branch's input
        has EXACTLY ZERO effect on the fused embedding.
        """
        gmu = GatedMultimodalUnit(latent_dim=self.latent_dim)
        x_img = torch.randn(self.batch_size, self.latent_dim)
        x_cbc_1 = torch.randn(self.batch_size, self.latent_dim)
        x_cbc_2 = torch.randn(self.batch_size, self.latent_dim) * 100.0 + 50.0  # Drastically different noise

        mask_img = torch.ones(self.batch_size, 1)
        mask_cbc = torch.zeros(self.batch_size, 1)  # CBC is missing

        fused_1, _ = gmu(x_img, x_cbc_1, mask_img=mask_img, mask_cbc=mask_cbc)
        fused_2, _ = gmu(x_img, x_cbc_2, mask_img=mask_img, mask_cbc=mask_cbc)

        # The two fused outputs must be completely identical
        self.assertTrue(torch.allclose(fused_1, fused_2, atol=1e-7),
                        "Missing CBC branch must not influence fused representation.")

    def test_gmu_both_modalities_missing_rejects_inference(self):
        """When both are missing (mask_img=0, mask_cbc=0), inference must be rejected with ValueError."""
        gmu = GatedMultimodalUnit(latent_dim=self.latent_dim)
        x_img = torch.randn(self.batch_size, self.latent_dim)
        x_cbc = torch.randn(self.batch_size, self.latent_dim)

        mask_img = torch.zeros(self.batch_size, 1)
        mask_cbc = torch.zeros(self.batch_size, 1)

        with self.assertRaises(ValueError):
            gmu(x_img, x_cbc, mask_img=mask_img, mask_cbc=mask_cbc)

    def test_end_to_end_classifier_rejects_both_missing(self):
        model = GMUAnemiaClassifier(cbc_features=self.cbc_features, latent_dim=self.latent_dim)
        mask_img = torch.zeros(self.batch_size, 1)
        mask_cbc = torch.zeros(self.batch_size, 1)

        with self.assertRaises(ValueError):
            model(self.precomputed_features, self.cbc_data, mask_img=mask_img, mask_cbc=mask_cbc)

    def test_end_to_end_classifier_forward_and_backward(self):
        model = GMUAnemiaClassifier(cbc_features=self.cbc_features, latent_dim=self.latent_dim)
        model.train()

        out = model(self.precomputed_features, self.cbc_data)

        self.assertIn("logits", out)
        self.assertIn("probability", out)
        self.assertIn("latent_gate_routing", out)
        self.assertIn("latent_gate_mean", out)
        self.assertIn("fused_embedding", out)

        self.assertEqual(out["logits"].shape, (self.batch_size, 1))
        self.assertEqual(out["probability"].shape, (self.batch_size, 1))
        self.assertEqual(out["latent_gate_routing"].shape, (self.batch_size, self.latent_dim))
        self.assertEqual(out["fused_embedding"].shape, (self.batch_size, self.latent_dim))

        probs = out["probability"]
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

        # Test backward pass
        targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        criterion = nn.BCELoss()
        loss = criterion(probs, targets)
        loss.backward()

        # Check gradients exist on classifier head
        grad_found = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.classifier.parameters())
        self.assertTrue(grad_found, "Gradients should propagate to classifier parameters.")


if __name__ == "__main__":
    unittest.main()
