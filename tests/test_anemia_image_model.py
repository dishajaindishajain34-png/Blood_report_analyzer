"""
test_anemia_image_model.py

Unit tests for image-based anemia model loading, preprocessing, and inference.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
from pathlib import Path

from src.anemia_image_model import predict_image_anemia, _preprocess_image


class TestAnemiaImageModel(unittest.TestCase):
    def setUp(self):
        # Create a dummy PIL image for testing
        self.dummy_img = Image.new("RGB", (300, 300), color=(100, 150, 200))

    @patch("src.anemia_image_model.WEIGHTS_PATH")
    def test_predict_image_anemia_missing_model(self, mock_path):
        # Force the model path to a non-existent one
        mock_path.exists.return_value = False
        res = predict_image_anemia(self.dummy_img)
        self.assertEqual(res["status"], "model_missing")
        self.assertEqual(res["prediction"], "Undetermined")
        self.assertEqual(res["probability"], 0.0)

    @patch("src.anemia_image_model._load_image_model")
    def test_predict_image_anemia_success(self, mock_load):
        # Mock PyTorch model outputs
        mock_model = MagicMock()
        import numpy as np
        
        # Mock model return logits (2-class output: [Normal_logits, Anemic_logits])
        # Say, Normal prob is 0.1, Anemic prob is 0.9
        import torch
        mock_logits = torch.tensor([[1.0, 3.2]]) # Softmax will give ~0.9 P(Anemic)
        mock_model.return_value = mock_logits
        
        mock_load.return_value = (mock_model, {"model_version": "1.0.0", "model_name": "Mock MobileNet"})
        
        with patch("src.anemia_image_model._preprocess_image") as mock_prep:
            mock_prep.return_value = torch.zeros((1, 3, 224, 224))
            res = predict_image_anemia(self.dummy_img)
            
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["prediction"], "Anemic")
            self.assertGreater(res["probability"], 0.5)
            self.assertEqual(res["model_version"], "1.0.0")

if __name__ == "__main__":
    unittest.main()
