import unittest

import numpy as np
import pandas as pd

from soc_baseline.modeling import evaluate_predictions


class ModelingTests(unittest.TestCase):
    def test_evaluate_predictions_computes_auc_with_business_label_order(self):
        y_true = pd.Series(["benign", "malicious", "suspicious", "benign", "malicious", "suspicious"])
        y_pred = y_true.copy()
        classes = ["benign", "malicious", "suspicious"]
        probabilities = np.array(
            [
                [0.98, 0.01, 0.01],
                [0.01, 0.98, 0.01],
                [0.01, 0.01, 0.98],
                [0.97, 0.02, 0.01],
                [0.02, 0.96, 0.02],
                [0.02, 0.01, 0.97],
            ]
        )

        metrics = evaluate_predictions(y_true, y_pred, probabilities, classes)

        self.assertEqual(metrics["macro_roc_auc_ovr"], 1.0)


if __name__ == "__main__":
    unittest.main()

