import unittest

import pandas as pd

from soc_baseline.data import detect_label_column, stratified_sample


class DataTests(unittest.TestCase):
    def test_detect_label_column_prefers_competition_label(self):
        df = pd.DataFrame({"event_id": ["1"], "label_binary": ["benign"]})
        self.assertEqual(detect_label_column(df), "label_binary")

    def test_stratified_sample_preserves_all_classes_when_possible(self):
        df = pd.DataFrame(
            {
                "event_id": [f"e{i}" for i in range(12)],
                "label_binary": ["benign"] * 8 + ["malicious"] * 2 + ["suspicious"] * 2,
            }
        )

        sample = stratified_sample(df, "label_binary", max_rows=6, random_state=7)

        self.assertEqual(len(sample), 6)
        self.assertEqual(set(sample["label_binary"]), {"benign", "malicious", "suspicious"})


if __name__ == "__main__":
    unittest.main()

