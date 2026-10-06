import tempfile
import unittest
from pathlib import Path

import pandas as pd

from soc_baseline.constants import LABELS
from soc_baseline.submission import validate_submission_frame, write_submission


class SubmissionTests(unittest.TestCase):
    def test_validate_submission_rejects_missing_extra_and_bad_labels(self):
        expected = pd.Series(["e1", "e2"])
        bad = pd.DataFrame(
            {
                "event_id": ["e1", "e3"],
                "pred_label": ["benign", "unknown"],
            }
        )

        with self.assertRaises(ValueError) as ctx:
            validate_submission_frame(bad, expected, LABELS)

        message = str(ctx.exception)
        self.assertIn("missing event_id", message)
        self.assertIn("extra event_id", message)
        self.assertIn("invalid pred_label", message)

    def test_write_submission_round_trips_expected_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "res.csv"
            written = write_submission(["e1", "e2"], ["benign", "malicious"], path)
            df = pd.read_csv(written)

        self.assertEqual(list(df.columns), ["event_id", "pred_label"])
        self.assertEqual(df.to_dict("records"), [
            {"event_id": "e1", "pred_label": "benign"},
            {"event_id": "e2", "pred_label": "malicious"},
        ])


if __name__ == "__main__":
    unittest.main()

