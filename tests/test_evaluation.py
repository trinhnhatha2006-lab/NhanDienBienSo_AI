from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_detector import choose_threshold, match_counts


class EvaluationTests(unittest.TestCase):
    def test_prefers_threshold_meeting_precision_and_recall_target(self) -> None:
        rows = [
            {"threshold": 0.25, "precision": 0.89, "recall": 1.0, "f1": 0.94},
            {"threshold": 0.50, "precision": 0.91, "recall": 0.92, "f1": 0.915},
        ]
        self.assertEqual(choose_threshold(rows)["threshold"], 0.50)

    def test_matches_each_ground_truth_at_most_once(self) -> None:
        predicted = np.asarray(
            [[0, 0, 100, 100], [2, 2, 98, 98]], dtype=np.float32
        )
        scores = np.asarray([0.9, 0.8], dtype=np.float32)
        ground_truth = np.asarray([[0, 0, 100, 100]], dtype=np.float32)

        self.assertEqual(match_counts(predicted, scores, ground_truth, 0.5), (1, 1, 0))


if __name__ == "__main__":
    unittest.main()
