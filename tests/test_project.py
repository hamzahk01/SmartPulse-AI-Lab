"""Unit and integration tests for SmartPulse AI Lab."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.city_router import DEPOT, HOSPITAL, OBSTACLES, PATIENT, run_dispatch_scenario
from src.download_data import download_kaggle_dataset
from src.knn_model import load_knn_bundle, train_and_evaluate_knn

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestSmartPulsePipeline(unittest.TestCase):
    """Verify dataset acquisition, KNN accuracy, A* optimality, and PDF report."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.csv_path = download_kaggle_dataset(PROJECT_ROOT / "data", force=False)
        cls.model_path = PROJECT_ROOT / "models" / "knn_traffic_model.pkl"
        cls.plot_path = PROJECT_ROOT / "outputs" / "knn_evaluation.png"
        if cls.model_path.exists():
            cls.bundle = load_knn_bundle(cls.model_path)
        else:
            cls.bundle = train_and_evaluate_knn(cls.csv_path, cls.model_path, cls.plot_path)

    def test_dataset_present_and_populated(self) -> None:
        self.assertTrue(self.csv_path.exists())
        self.assertGreaterEqual(self.bundle["metrics"]["dataset_rows"], 5000)

    def test_knn_model_accuracy(self) -> None:
        acc = self.bundle["metrics"]["test_accuracy"]
        self.assertGreaterEqual(acc, 0.85, f"Expected KNN accuracy >= 85%, got {acc:.2%}")

    def test_astar_beats_or_ties_bfs_cost(self) -> None:
        rush = run_dispatch_scenario(
            self.bundle, name="Rush Hour Test", hour=8, day_name="Monday"
        )
        self.assertLessEqual(rush["astar"].total_cost, rush["bfs"].total_cost)
        self.assertEqual(rush["astar"].full_path[0], DEPOT)
        self.assertIn(PATIENT, rush["astar"].full_path)
        self.assertEqual(rush["astar"].full_path[-1], HOSPITAL)

        # Ensure neither route collides with building obstacles
        for cell in rush["astar"].full_path:
            self.assertNotIn(cell, OBSTACLES)
        for cell in rush["bfs"].full_path:
            self.assertNotIn(cell, OBSTACLES)

    def test_pdf_report_generated(self) -> None:
        pdf_path = PROJECT_ROOT / "SmartPulse_AI_Project_Report.pdf"
        self.assertTrue(pdf_path.exists(), "PDF report should exist after running main.py")
        self.assertGreater(pdf_path.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
