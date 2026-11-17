"""Check the published synthetic run's output contracts, without downloading weights."""

import csv
import json
import unittest
from pathlib import Path

import numpy as np

from variant_prediction import regression_metrics

ROOT = Path(__file__).resolve().parents[1]


class SavedResultTests(unittest.TestCase):
    def test_validation_metrics_match_exported_predictions(self):
        metrics = json.loads((ROOT / "results/synthetic/metrics.json").read_text())
        with (ROOT / "results/synthetic/validation.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        actual = regression_metrics(
            np.array([float(r["DMS_score"]) for r in rows], dtype=np.float32),
            np.array([float(r["DMS_score_predicted"]) for r in rows], dtype=np.float32))
        self.assertEqual(len(rows), metrics["validation_variants"])
        self.assertAlmostEqual(actual["mse"], metrics["validation"]["mse"], places=6)
        self.assertAlmostEqual(actual["spearman"], metrics["validation"]["spearman"], places=6)
        self.assertFalse(set(metrics["train_positions"]) & set(metrics["validation_positions"]))
        self.assertEqual({int(r["position"]) for r in rows}, set(metrics["validation_positions"]))

    def test_ranking_matches_candidate_predictions(self):
        with (ROOT / "results/synthetic/predictions.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        with (ROOT / "examples/synthetic/candidates.csv").open() as handle:
            candidates = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 20)
        self.assertEqual({r["mutant"] for r in rows}, {r["mutant"] for r in candidates})
        self.assertTrue(all(np.isfinite(float(r["DMS_score_predicted"])) for r in rows))
        ranking = sorted(rows, key=lambda r: (-float(r["DMS_score_predicted"]), r["mutant"]))
        self.assertEqual((ROOT / "results/synthetic/top10.txt").read_text().splitlines(),
                         [r["mutant"] for r in ranking[:10]])


if __name__ == "__main__":
    unittest.main()
