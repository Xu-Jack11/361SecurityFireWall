import unittest

import numpy as np
import pandas as pd

from soc_baseline.source_mask import (
    allowed_label_matrix,
    fit_source_label_mask,
    predict_with_source_mask,
    source_keys,
)

CLASSES = ["benign", "suspicious", "malicious"]


class _FixedProbaModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, documents):
        return self.probabilities


class SourceMaskTests(unittest.TestCase):
    def test_source_keys_join_vendor_and_product_and_handle_missing(self):
        frame = pd.DataFrame({"vendor_name": ["Cisco", None], "product_name": ["ASA Firewall", None]})
        self.assertEqual(source_keys(frame).tolist(), ["Cisco/ASA Firewall", "/"])

    def test_fit_keeps_only_supported_sources_and_observed_labels(self):
        sources = pd.Series(["aws"] * 3 + ["asa"] * 3 + ["rare"])
        labels = pd.Series(["benign"] * 3 + ["suspicious", "suspicious", "benign"] + ["malicious"])

        mask = fit_source_label_mask(sources, labels, min_rows=3)

        self.assertEqual(mask, {"aws": ["benign"], "asa": ["benign", "suspicious"]})

    def test_allowed_matrix_leaves_unknown_sources_unrestricted(self):
        allowed = allowed_label_matrix(pd.Series(["aws", "new"]), {"aws": ["benign"]}, CLASSES)
        np.testing.assert_array_equal(allowed, [[True, False, False], [True, True, True]])

    def test_predict_picks_best_allowed_label(self):
        model = _FixedProbaModel([[0.2, 0.1, 0.7], [0.2, 0.1, 0.7], [0.1, 0.3, 0.6]])
        sources = pd.Series(["aws", "new", "asa"])
        mask = {"aws": ["benign"], "asa": ["benign", "suspicious"]}

        predictions = predict_with_source_mask(model, pd.Series(["", "", ""]), sources, mask, CLASSES)

        self.assertEqual(predictions.tolist(), ["benign", "malicious", "suspicious"])

    def test_predict_falls_back_when_no_model_class_is_allowed(self):
        model = _FixedProbaModel([[0.3, 0.7]])
        predictions = predict_with_source_mask(
            model, pd.Series([""]), pd.Series(["x"]), {"x": ["malicious"]}, ["benign", "suspicious"]
        )
        self.assertEqual(predictions.tolist(), ["suspicious"])


if __name__ == "__main__":
    unittest.main()
