"""
test_gmu_integration_e2e.py

End-to-end integration and smoke tests for the selectable Experimental GMU Fusion mode
in the FastAPI backend, inference pipeline, and PDF report generator.
"""

from __future__ import annotations

import io
import os
import unittest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app
from src.anemia_fusion import predict_gmu_experimental, predict_fusion
from src.report_generator import generate_report
from src.normal_ranges import AnalysisFlags


class TestGMUIntegrationE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

        # Create a small valid test JPEG smear image in-memory
        img = Image.new("RGB", (224, 224), color=(140, 70, 70))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        cls.test_image_bytes = buf.getvalue()
        cls.test_pil = img

    def test_01_default_fusion_preserves_production_model(self):
        """Confirm default mode runs stacking ensemble / default fusion and is NOT experimental."""
        files = {"file": ("test_smear.jpg", self.test_image_bytes, "image/jpeg")}
        data = {
            "gender": "Female",
            "hb": 11.2,
            "mcv": 78.0,
            "mch": 25.0,
            "mchc": 31.0,
            "fusion_mode": "default",
        }
        resp = self.client.post("/analyze", files=files, data=data)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()

        self.assertIn("anemia_fusion", body)
        fusion = body["anemia_fusion"]
        self.assertEqual(fusion["status"], "success")
        self.assertEqual(fusion.get("fusion_mode"), "default")
        self.assertFalse(fusion.get("is_experimental", True))
        self.assertNotEqual(fusion["fusion_strategy"], "experimental_gmu_dual")

    def test_02_experimental_gmu_dual_modality_route(self):
        """Confirm /analyze routes to GMU when fusion_mode='experimental_gmu'."""
        files = {"file": ("test_smear.jpg", self.test_image_bytes, "image/jpeg")}
        data = {
            "gender": "Female",
            "hb": 8.5,
            "mcv": 69.0,
            "mch": 22.0,
            "mchc": 28.5,
            "fusion_mode": "experimental_gmu",
        }
        resp = self.client.post("/analyze", files=files, data=data)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()

        self.assertIn("anemia_fusion", body)
        fusion = body["anemia_fusion"]
        self.assertEqual(fusion["status"], "success")
        self.assertEqual(fusion["fusion_mode"], "experimental_gmu")
        self.assertTrue(fusion["is_experimental"])
        self.assertEqual(fusion["fusion_strategy"], "experimental_gmu_dual")
        self.assertIsNotNone(fusion.get("latent_gate_mean"))
        self.assertIn("Academic Prototype Notice", fusion.get("limitation_notice", ""))
        self.assertIn("Anemic", fusion["fusion_prediction"])

    def test_03_experimental_gmu_image_only_route(self):
        """Confirm /analyze handles experimental GMU with missing CBC (image-only)."""
        files = {"file": ("test_smear.jpg", self.test_image_bytes, "image/jpeg")}
        data = {
            "fusion_mode": "experimental_gmu",
        }
        resp = self.client.post("/analyze", files=files, data=data)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()

        fusion = body["anemia_fusion"]
        self.assertEqual(fusion["status"], "success")
        self.assertEqual(fusion["fusion_strategy"], "experimental_gmu_image_only")
        self.assertTrue(fusion["is_experimental"])

    def test_04_direct_predict_gmu_cbc_only(self):
        """Confirm predict_gmu_experimental operates in cbc-only mode when image is missing."""
        cbc_payload = {
            "Gender": 0,
            "Hemoglobin": 9.0,
            "MCV": 72.0,
            "MCH": 23.0,
            "MCHC": 29.0,
        }
        res = predict_gmu_experimental(image=None, cbc_params=cbc_payload)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["fusion_strategy"], "experimental_gmu_cbc_only")
        self.assertTrue(res["is_experimental"])
        self.assertEqual(res["latent_gate_mean"], 0.0)

    def test_05_direct_predict_gmu_neither_input_insufficient_data(self):
        """Confirm predict_gmu_experimental safely rejects when both modalities are absent."""
        res = predict_gmu_experimental(image=None, cbc_params=None)
        self.assertEqual(res["status"], "insufficient_data")
        self.assertEqual(res["fusion_prediction"], "Undetermined")
        self.assertEqual(res["fusion_probability"], 0.0)
        self.assertTrue(res["is_experimental"])

    def test_06_pdf_report_rendering_with_gmu_result(self):
        """Verify PDF generation executes cleanly and renders the experimental GMU notice."""
        gmu_res = {
            "fusion_prediction": "Anemic",
            "is_anemic": True,
            "fusion_probability": 0.892,
            "fusion_confidence": 0.892,
            "fusion_strategy": "experimental_gmu_dual",
            "latent_gate_mean": 0.435,
            "model_version": "1.0.0-experimental-trained",
            "status": "success",
            "message": "Experimental GMU inference executed.",
            "fusion_mode": "experimental_gmu",
            "is_experimental": True,
            "limitation_notice": "Academic Prototype Notice: Trained on synthetic image-CBC pairings.",
        }

        flags = AnalysisFlags(
            cell_flags=[],
            diff_flags=[],
            overall="Abnormal",
            summary=["[!] Anemia indicators present"],
        )

        pdf_path = generate_report(
            analysis_id="TEST-GMU-REPORT-01",
            original_image=self.test_pil,
            annotated_image=self.test_pil,
            raw_counts={"rbc": 20, "wbc": 2, "platelet": 4},
            calibrated_counts={"rbc": 4.5, "wbc": 6.0, "platelet": 250},
            wbc_differential={"Neutrophil": 60.0, "Lymphocyte": 30.0, "Monocyte": 5.0, "Eosinophil": 4.0, "Basophil": 1.0},
            flags=flags,
            avg_confidence=0.88,
            anemia_fusion_result=gmu_res,
        )

        self.assertIsNotNone(pdf_path)
        p = Path(pdf_path)
        self.assertTrue(p.exists())
        self.assertGreater(p.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
