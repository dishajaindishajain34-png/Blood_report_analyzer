"""
test_gmu_experimental_inference.py

Unit tests for the experimental GMU trained checkpoint loading, preprocessing,
and inference across all modality configurations.
"""

from __future__ import annotations

import unittest
from PIL import Image
import torch

from src.anemia_fusion import predict_gmu_experimental, _load_gmu_experimental


class TestGMUExperimentalInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dummy_img = Image.new("RGB", (224, 224), color=(120, 80, 80))
        cls.normal_cbc = {
            "Gender": 1,
            "Hemoglobin": 15.5,
            "MCV": 90.0,
            "MCH": 30.0,
            "MCHC": 34.0,
        }
        cls.anemic_cbc = {
            "Gender": 0,
            "Hemoglobin": 8.5,
            "MCV": 70.0,
            "MCH": 22.0,
            "MCHC": 29.0,
        }

    def test_01_load_experimental_gmu_checkpoint(self):
        """Verify the trained experimental GMU checkpoint loads cleanly."""
        model, scaler, meta = _load_gmu_experimental()
        self.assertIsNotNone(model, "GMU model should not be None")
        self.assertIsNotNone(scaler, "CBC scaler should not be None")
        self.assertEqual(meta.get("status"), "EXPERIMENTAL_SYNTHETIC_DATA")

    def test_02_dual_modality_anemic_inference(self):
        """Verify dual-modality inference on an anemic CBC profile."""
        res = predict_gmu_experimental(image=self.dummy_img, cbc_params=self.anemic_cbc)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["fusion_strategy"], "experimental_gmu_dual")
        self.assertIn("fusion_probability", res)
        self.assertIn("latent_gate_mean", res)
        self.assertGreaterEqual(res["fusion_probability"], 0.0)
        self.assertLessEqual(res["fusion_probability"], 1.0)

    def test_03_dual_modality_normal_inference(self):
        """Verify dual-modality inference on a normal CBC profile."""
        res = predict_gmu_experimental(image=self.dummy_img, cbc_params=self.normal_cbc)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["fusion_strategy"], "experimental_gmu_dual")
        self.assertIn("fusion_probability", res)

    def test_04_image_only_inference(self):
        """Verify inference when CBC is missing (mask_img=1, mask_cbc=0)."""
        res = predict_gmu_experimental(image=self.dummy_img, cbc_params=None)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["fusion_strategy"], "experimental_gmu_image_only")
        self.assertIn("fusion_probability", res)

    def test_05_cbc_only_inference(self):
        """Verify inference when image is missing (mask_img=0, mask_cbc=1)."""
        res = predict_gmu_experimental(image=None, cbc_params=self.anemic_cbc)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["fusion_strategy"], "experimental_gmu_cbc_only")
        self.assertIn("fusion_probability", res)

    def test_06_neither_modality_rejection(self):
        """Verify inference rejection when both modalities are missing."""
        res = predict_gmu_experimental(image=None, cbc_params=None)
        self.assertEqual(res["status"], "insufficient_data")
        self.assertEqual(res["fusion_prediction"], "Undetermined")


if __name__ == "__main__":
    unittest.main()
