"""
test_cell_detection.py

Unit tests for cell detection pipeline (detect_cells, _detect_cells_tiled).
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

from src.detect import detect_cells, _detect_cells_tiled, CLASS_NAMES


class TestCellDetection(unittest.TestCase):
    def setUp(self):
        # Create a blank RGB image (small enough to not trigger tiled fallback)
        self.dummy_pil = Image.new("RGB", (300, 300), color=(255, 255, 255))
        self.dummy_np = np.array(self.dummy_pil)

    @patch("src.detect._load_model")
    def test_detect_cells_pil_rgb_input(self, mock_load):
        # Set up a mock YOLO model
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        # Setup mock results
        mock_box = MagicMock()
        
        mock_cls = MagicMock()
        mock_cls.__getitem__.return_value = mock_cls
        mock_cls.item.return_value = 1.0  # RBC
        mock_box.cls = mock_cls

        mock_conf = MagicMock()
        mock_conf.__getitem__.return_value = mock_conf
        mock_conf.item.return_value = 0.85
        mock_box.conf = mock_conf

        mock_xyxy = MagicMock()
        mock_xyxy.__getitem__.return_value = mock_xyxy
        mock_xyxy.tolist.return_value = [10.0, 20.0, 50.0, 60.0]
        mock_box.xyxy = mock_xyxy

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.plot.return_value = np.zeros((640, 640, 3), dtype=np.uint8)

        mock_model.predict.return_value = [mock_result]

        # Call detect_cells
        res = detect_cells(self.dummy_pil, conf_threshold=0.15, model_path="dummy_path")

        # Verify model was called with PIL image directly (to prevent RGB/BGR bug)
        mock_model.predict.assert_called_once()
        called_args, called_kwargs = mock_model.predict.call_args
        self.assertIsInstance(called_kwargs["source"], Image.Image)

        # Check counts
        self.assertEqual(res.raw_counts["rbc"], 1)
        self.assertEqual(res.raw_counts["wbc"], 0)
        self.assertEqual(res.raw_counts["platelet"], 0)

    @patch("src.detect._load_model")
    def test_detect_cells_numpy_rgb_input(self, mock_load):
        # Set up a mock YOLO model
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        # Setup mock results
        mock_box = MagicMock()
        
        mock_cls = MagicMock()
        mock_cls.__getitem__.return_value = mock_cls
        mock_cls.item.return_value = 2.0  # WBC
        mock_box.cls = mock_cls

        mock_conf = MagicMock()
        mock_conf.__getitem__.return_value = mock_conf
        mock_conf.item.return_value = 0.90
        mock_box.conf = mock_conf

        mock_xyxy = MagicMock()
        mock_xyxy.__getitem__.return_value = mock_xyxy
        mock_xyxy.tolist.return_value = [100.0, 100.0, 200.0, 200.0]
        mock_box.xyxy = mock_xyxy

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.plot.return_value = np.zeros((640, 640, 3), dtype=np.uint8)

        mock_model.predict.return_value = [mock_result]

        # Call detect_cells with NumPy RGB array
        res = detect_cells(self.dummy_np, conf_threshold=0.15, model_path="dummy_path")

        # Verify model was called with converted BGR NumPy array
        mock_model.predict.assert_called_once()
        called_args, called_kwargs = mock_model.predict.call_args
        self.assertIsInstance(called_kwargs["source"], np.ndarray)

        # Check counts
        self.assertEqual(res.raw_counts["rbc"], 0)
        self.assertEqual(res.raw_counts["wbc"], 1)
        self.assertEqual(res.raw_counts["platelet"], 0)

    @patch("src.detect._detect_cells_tiled")
    @patch("src.detect._load_model")
    def test_tiled_detection_fallback(self, mock_load, mock_tiled):
        # Mock normal predict showing 0 RBCs
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        mock_result = MagicMock()
        mock_result.boxes = []  # No detections
        mock_result.plot.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
        mock_model.predict.return_value = [mock_result]

        # Mock tiled detection response
        mock_tiled.return_value = (
            {"rbc": 12, "wbc": 1, "platelet": 2},
            [{"class": "rbc", "confidence": 0.8, "bbox": [0,0,10,10]}],
            np.zeros((640, 640, 3), dtype=np.uint8)
        )

        large_pil = Image.new("RGB", (640, 640), color=(255, 255, 255))
        res = detect_cells(large_pil, conf_threshold=0.15, model_path="dummy_path")

        # Verify tiled fallback was invoked
        mock_tiled.assert_called_once()
        self.assertEqual(res.raw_counts["rbc"], 12)
        self.assertEqual(res.raw_counts["wbc"], 1)
        self.assertEqual(res.raw_counts["platelet"], 2)


if __name__ == "__main__":
    unittest.main()
