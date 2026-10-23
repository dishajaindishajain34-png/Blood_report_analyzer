"""
test_anemia_fusion.py

Unit tests for multimodal fusion anemia prediction.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.anemia_fusion import predict_fusion


class TestAnemiaFusion(unittest.TestCase):
    def test_predict_fusion_no_inputs(self):
        res = predict_fusion(None, None)
        self.assertEqual(res["status"], "insufficient_data")
        self.assertEqual(res["fusion_prediction"], "Undetermined")

    def test_predict_fusion_image_only(self):
        # 0.8 probability -> Anemic
        res = predict_fusion(0.8, None)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["fusion_strategy"], "image_only")
        self.assertEqual(res["fusion_prediction"], "Anemic")
        self.assertEqual(res["fusion_probability"], 0.8)

        # 0.2 probability -> Normal
        res2 = predict_fusion(0.2, None)
        self.assertEqual(res2["fusion_prediction"], "Normal")

    def test_predict_fusion_cbc_only(self):
        res = predict_fusion(None, 0.9)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["fusion_strategy"], "cbc_only")
        self.assertEqual(res["fusion_prediction"], "Anemic")
        self.assertEqual(res["fusion_probability"], 0.9)

    def test_predict_fusion_average_fallback(self):
        # Model is not loaded (mock no file)
        with patch("src.anemia_fusion.FUSION_MODEL_PATH") as mock_path:
            mock_path.exists.return_value = False
            res = predict_fusion(0.8, 0.2)
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["fusion_strategy"], "average_fallback")
            # (0.8 + 0.2)/2 = 0.5 -> Normal (since threshold is 0.5)
            self.assertEqual(res["fusion_probability"], 0.5)
            self.assertEqual(res["fusion_prediction"], "Normal")

            res2 = predict_fusion(0.9, 0.3)
            # (0.9 + 0.3)/2 = 0.6 -> Anemic
            self.assertEqual(res2["fusion_probability"], 0.6)
            self.assertEqual(res2["fusion_prediction"], "Anemic")

    @patch("src.anemia_fusion._load_fusion_model")
    def test_predict_fusion_learned(self, mock_load):
        # Mock Logistic Regression model
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]  # 1 = Anemic
        mock_model.predict_proba.return_value = [[0.15, 0.85]]
        
        mock_load.return_value = (mock_model, {"model_version": "1.0.0"})
        
        res = predict_fusion(0.7, 0.8)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["fusion_strategy"], "learned_logistic_regression")
        self.assertEqual(res["fusion_prediction"], "Anemic")
        self.assertEqual(res["fusion_probability"], 0.85)
        # Verify predict_proba was actually called
        mock_model.predict_proba.assert_called_once()

    def test_real_logistic_regression_joblib_compatibility(self):
        import tempfile
        import joblib
        from pathlib import Path
        from sklearn.linear_model import LogisticRegression

        # Create a real LogisticRegression model trained on synthetic 2-feature inputs
        lr = LogisticRegression()
        X_dummy = [[0.1, 0.2], [0.9, 0.8], [0.3, 0.4], [0.85, 0.95]]
        y_dummy = [0, 1, 0, 1]
        lr.fit(X_dummy, y_dummy)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "fusion_model.joblib"
            joblib.dump(lr, tmp_path)

            with patch("src.anemia_fusion.FUSION_MODEL_PATH", tmp_path), \
                 patch("src.anemia_fusion._cached_fusion", None):
                res = predict_fusion(0.9928, 0.85)
                self.assertEqual(res["status"], "success")
                self.assertEqual(res["fusion_strategy"], "learned_logistic_regression")
                self.assertIn(res["fusion_prediction"], ["Anemic", "Normal"])
                self.assertTrue(0.0 <= res["fusion_probability"] <= 1.0)
                # Verify probability came from predict_proba, not (0.9928 + 0.85)/2 = 0.9214
                prob_lr = float(lr.predict_proba([[0.9928, 0.85]])[0][1])
                self.assertAlmostEqual(res["fusion_probability"], round(prob_lr, 4))

    def test_trained_stacking_ensemble_live(self):
        # Test real trained model from models/anemia_fusion/
        from src.anemia_fusion import FUSION_MODEL_PATH
        if not FUSION_MODEL_PATH.exists():
            self.skipTest("Trained fusion model not found on disk")

        # 1. Normal patient case with noisy image
        normal_cbc = {"Gender": 1, "Hemoglobin": 14.5, "MCV": 88.0, "MCH": 29.5, "MCHC": 33.5}
        res_norm = predict_fusion(0.75, 0.01, cbc_params=normal_cbc)
        self.assertEqual(res_norm["status"], "success")
        self.assertEqual(res_norm["fusion_strategy"], "learned_stacking_ensemble")
        self.assertEqual(res_norm["fusion_prediction"], "Normal")
        self.assertLess(res_norm["fusion_probability"], 0.20)

        # 2. Anemic patient case
        anemic_cbc = {"Gender": 0, "Hemoglobin": 8.5, "MCV": 74.0, "MCH": 23.0, "MCHC": 29.5}
        res_anem = predict_fusion(0.92, 0.98, cbc_params=anemic_cbc)
        self.assertEqual(res_anem["status"], "success")
        self.assertEqual(res_anem["fusion_strategy"], "learned_stacking_ensemble")
        self.assertEqual(res_anem["fusion_prediction"], "Anemic")
        self.assertGreater(res_anem["fusion_probability"], 0.85)

if __name__ == "__main__":
    unittest.main()
