import unittest

import numpy as np

from soc_baseline.decision import weighted_argmax

CLASSES = ["benign", "suspicious", "malicious"]


class WeightedArgmaxTests(unittest.TestCase):
    def test_without_weights_is_plain_argmax(self):
        probabilities = np.array([[0.2, 0.7, 0.1], [0.6, 0.3, 0.1]])
        self.assertEqual(weighted_argmax(probabilities, CLASSES).tolist(), ["suspicious", "benign"])

    def test_benign_weight_requires_alert_to_beat_weighted_benign(self):
        probabilities = np.array([[0.2, 0.8, 0.0], [0.05, 0.95, 0.0]])
        predictions = weighted_argmax(probabilities, CLASSES, label_weights={"benign": 10})
        # 0.8 < 10 * 0.2 -> benign; 0.95 > 10 * 0.05 -> suspicious
        self.assertEqual(predictions.tolist(), ["benign", "suspicious"])

    def test_disallowed_labels_are_never_chosen_even_with_weight(self):
        probabilities = np.array([[0.3, 0.1, 0.6]])
        allowed = np.array([[False, True, True]])
        predictions = weighted_argmax(probabilities, CLASSES, allowed, {"benign": 100})
        self.assertEqual(predictions.tolist(), ["malicious"])

    def test_empty_allowed_row_falls_back_to_all_labels(self):
        probabilities = np.array([[0.3, 0.6, 0.1]])
        allowed = np.array([[False, False, False]])
        self.assertEqual(weighted_argmax(probabilities, CLASSES, allowed).tolist(), ["suspicious"])

    def test_does_not_mutate_inputs(self):
        probabilities = np.array([[0.5, 0.5, 0.0]])
        allowed = np.array([[False, False, False]])
        weighted_argmax(probabilities, CLASSES, allowed, {"benign": 2})
        np.testing.assert_array_equal(probabilities, [[0.5, 0.5, 0.0]])
        np.testing.assert_array_equal(allowed, [[False, False, False]])


if __name__ == "__main__":
    unittest.main()
