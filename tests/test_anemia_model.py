"""
test_anemia_model.py — Unit and Integration Tests for ML Anemia Detection & Dual-Stream Pipeline.
"""

import sys
import unittest
from pathlib import Path
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.anemia_model import predict_anemia_ml, load_anemia_pipeline
from src.normal_ranges import run_full_analysis
from src.report_generator import generate_report
from database.connection import get_session_factory, init_db
from database.crud import create_record, get_record_by_analysis_id, delete_record


class TestAnemiaMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize database for test execution."""
        init_db()
        cls.SessionLocal = get_session_factory()

    def test_01_model_artifacts_exist_and_load(self):
        """Verify that ML Anemia model artifacts load cleanly."""
        pipeline = load_anemia_pipeline()
        self.assertIsNotNone(pipeline, "Failed to load ML Anemia pipeline artifacts.")
        model, scaler, feature_cols, metadata = pipeline
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)
        self.assertEqual(feature_cols, ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"])
        self.assertEqual(metadata.get("model_name"), "Random Forest")

    def test_02_predict_anemic_sample(self):
        """Test prediction for an explicit Anemic CBC profile (Female, Hb=9.2, MCV=71)."""
        input_data = {
            "Gender": "Female",
            "Hemoglobin": 9.2,
            "MCV": 71.0,
            "MCH": 21.0,
            "MCHC": 29.5
        }
        res = predict_anemia_ml(input_data)
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["is_anemic"])
        self.assertEqual(res["prediction"], "Anemic")
        self.assertGreater(res["probability"], 0.50)

    def test_03_predict_normal_sample(self):
        """Test prediction for an explicit Normal CBC profile (Male, Hb=15.5, MCV=89)."""
        input_data = {
            "Gender": "Male",
            "Hemoglobin": 15.5,
            "MCV": 89.0,
            "MCH": 29.5,
            "MCHC": 34.0
        }
        res = predict_anemia_ml(input_data)
        self.assertEqual(res["status"], "success")
        self.assertFalse(res["is_anemic"])
        self.assertEqual(res["prediction"], "Normal")
        self.assertLess(res["probability"], 0.50)

    def test_04_missing_input_validation(self):
        """Verify that missing required CBC fields return invalid_input status without crashing."""
        incomplete_data = {
            "Gender": "Female",
            "Hemoglobin": 11.5,
            # Missing MCV, MCH, MCHC
        }
        res = predict_anemia_ml(incomplete_data)
        self.assertEqual(res["status"], "invalid_input")
        self.assertEqual(res["prediction"], "Undetermined")
        self.assertIn("Missing required CBC parameter(s)", res["message"])

    def test_05_out_of_range_input_validation(self):
        """Verify that out-of-range physiological values are safely caught."""
        invalid_data = {
            "Gender": "Male",
            "Hemoglobin": -5.0, # Unphysiological
            "MCV": 88.0,
            "MCH": 28.0,
            "MCHC": 33.0
        }
        res = predict_anemia_ml(invalid_data)
        self.assertEqual(res["status"], "invalid_input")
        self.assertIn("out of physiological range", res["message"])

    def test_06_database_crud_with_ml_fields(self):
        """Test SQLite record creation and retrieval including secondary ML fields."""
        aid = "TEST-ANEMIA-ML-999"
        test_payload = {
            "analysis_id": aid,
            "patient_name": "Test Patient",
            "patient_id": "P-100",
            "patient_age": 35,
            "raw_rbc": 350, "raw_wbc": 10, "raw_platelet": 40,
            "cal_rbc": 4500000, "cal_wbc": 7000, "cal_platelet": 250000,
            "overall_result": "Normal",
            "anemia_ml_prediction": "Anemic",
            "anemia_ml_probability": 0.96,
            "anemia_ml_model_version": "1.0.0",
            "cbc_inputs_json": {"Gender": 0, "Hemoglobin": 9.5, "MCV": 72.0, "MCH": 21.0, "MCHC": 29.0}
        }
        with self.SessionLocal() as db:
            # Cleanup prior run if needed
            delete_record(db, aid)
            rec = create_record(db, test_payload)
            self.assertEqual(rec.analysis_id, aid)
            self.assertEqual(rec.anemia_ml_prediction, "Anemic")
            self.assertEqual(rec.anemia_ml_probability, 0.96)

            fetched = get_record_by_analysis_id(db, aid)
            self.assertIsNotNone(fetched)
            self.assertEqual(fetched.anemia_ml_prediction, "Anemic")

            # Cleanup
            delete_record(db, aid)

    def test_07_pdf_report_with_ml_section(self):
        """Test PDF report generation with ML Anemia Analysis block."""
        aid = "TEST-RPT-ML-888"
        orig = Image.new("RGB", (640, 480), color=(240, 240, 240))
        ann = Image.new("RGB", (640, 480), color=(220, 230, 255))
        raw = {"rbc": 300, "wbc": 10, "platelet": 35}
        cal = {"rbc": 4600000, "wbc": 7500, "platelet": 220000}
        diff = {"Neutrophil": 60.0, "Lymphocyte": 30.0, "Monocyte": 6.0, "Eosinophil": 3.0, "Basophil": 1.0}
        flags = run_full_analysis(cal, diff)

        ml_res = {
            "status": "success",
            "prediction": "Anemic",
            "is_anemic": True,
            "probability": 0.94,
            "confidence": 0.94,
            "model_name": "Random Forest",
            "model_version": "1.0.0"
        }

        out_path = generate_report(
            analysis_id=aid,
            original_image=orig,
            annotated_image=ann,
            raw_counts=raw,
            calibrated_counts=cal,
            wbc_differential=diff,
            flags=flags,
            avg_confidence=0.91,
            anemia_ml_result=ml_res,
            output_dir=PROJECT_ROOT / "reports"
        )
        self.assertTrue(out_path.exists())
        self.assertGreater(out_path.stat().st_size, 1000)

    def test_08_existing_blood_smear_pipeline_non_breaking(self):
        """Verify that existing detect_cells pipeline module is non-breaking."""
        from src.detect import detect_cells
        img = Image.new("RGB", (640, 480), color=(200, 200, 200))
        res = detect_cells(img, conf_threshold=0.35, iou_threshold=0.40)
        self.assertIsNotNone(res)
        self.assertIn("rbc", res.raw_counts)
        self.assertIn("wbc", res.raw_counts)
        self.assertIn("platelet", res.raw_counts)


if __name__ == "__main__":
    unittest.main()
