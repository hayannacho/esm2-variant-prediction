import json
import tempfile
import unittest
from pathlib import Path

import torch

from variant_prediction import (
    RegressionHead, mutate, read_fasta, read_variants, regression_metrics,
    split_positions, write_csv,
)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_zero_based_boundary_substitutions(self):
        self.assertEqual(mutate("MAC", "M0A"), ("AAC", 0))
        self.assertEqual(mutate("MAC", "C2W"), ("MAW", 2))

    def test_reject_invalid_mutations(self):
        for mutation in ("M1A", "C3W", "M-1A", "M00A", "M0M", "M0X", "m0a", "M0A:C2W", ""):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                mutate("MAC", mutation)

    def test_fasta_multiline(self):
        path = self.root / "input.fasta"
        path.write_text(">artificial\nma\nc\n")
        self.assertEqual(read_fasta(path), "MAC")

    def test_reject_multiple_or_invalid_fasta(self):
        path = self.root / "input.fasta"
        for content in (">one\nMA\n>two\nC\n", "MAC", ">one\n", ">one\nMAX", ">one\n" + "A" * 1023):
            with self.subTest(content=content[:20]), self.assertRaises(ValueError):
                path.write_text(content)
                read_fasta(path)

    def test_labeled_and_candidate_csv(self):
        path = self.root / "input.csv"
        write_csv(path, ["mutant", "DMS_score"], [{"mutant": "M0A", "DMS_score": 1.25}])
        result = read_variants(path, "MAC", True)[0]
        self.assertEqual(result["sequence"], "AAC")
        self.assertEqual(result["DMS_score"], 1.25)
        write_csv(path, ["mutant"], [{"mutant": "C2W"}])
        self.assertEqual(read_variants(path, "MAC", False)[0]["position"], 2)

    def test_reject_bad_scores_headers_empty_and_duplicates(self):
        path = self.root / "input.csv"
        inputs = ["mutant,DMS_score\n", "mutation,score\nM0A,1\n",
                  "mutant,DMS_score\nM0A,1\nM0A,2\n"]
        inputs += [f"mutant,DMS_score\nM0A,{score}\n" for score in ("nan", "inf", "-inf", "oops", "", "1e100")]
        for content in inputs:
            with self.subTest(content=content), self.assertRaises(ValueError):
                path.write_text(content)
                read_variants(path, "MAC", True)


class ModelTests(unittest.TestCase):
    def test_position_split_is_disjoint_complete_and_repeatable(self):
        rows = [{"position": p} for p in range(20) for _ in range(3)]
        train, valid = split_positions(rows, 0.2, 42)
        self.assertEqual((len(train), len(valid)), (48, 12))
        self.assertEqual(set(train) | set(valid), set(range(60)))
        self.assertFalse({rows[i]["position"] for i in train} & {rows[i]["position"] for i in valid})
        self.assertEqual((train, valid), split_positions(rows, 0.2, 42))

    def test_reject_impossible_split(self):
        for rows, fraction in [([{"position": 1}], 0.2), ([], 0.2), ([{"position": 1}, {"position": 2}], 1)]:
            with self.assertRaises(ValueError):
                split_positions(rows, fraction, 42)

    def test_regression_head_learns_from_embeddings(self):
        torch.manual_seed(7)
        inputs = torch.randn(20, 4)
        targets = inputs[:, 0] - inputs[:, 1]
        model = RegressionHead(4, 16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        initial = torch.nn.functional.mse_loss(model(inputs), targets).item()
        for _ in range(100):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
        self.assertEqual(model(inputs[:1]).shape, (1,))
        self.assertLess(torch.nn.functional.mse_loss(model(inputs), targets).item(), initial * 0.1)

    def test_metrics_and_undefined_correlation(self):
        result = regression_metrics([1, 2, 3], [3, 2, 1])
        self.assertAlmostEqual(result["mse"], 8 / 3)
        self.assertAlmostEqual(result["spearman"], -1)
        result = regression_metrics([1, 2, 3], [2, 2, 2])
        self.assertIsNone(result["spearman"])
        json.dumps(result, allow_nan=False)
        self.assertIsNone(regression_metrics([1], [2])["spearman"])


if __name__ == "__main__":
    unittest.main()
