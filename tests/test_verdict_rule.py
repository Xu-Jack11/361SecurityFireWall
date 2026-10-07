import unittest

import pandas as pd

from soc_baseline.verdict_rule import apply_verdict_rule, fit_verdict_rule, verdict_cells


def _frame(rows):
    return pd.DataFrame(rows, columns=["vendor_name", "message_sanitized"])


class VerdictRuleTests(unittest.TestCase):
    def test_cells_combine_vendor_state_and_verdict(self):
        frame = _frame(
            [
                (None, "Deny tcp src outside"),
                ("Cisco", "Deny tcp src outside"),
                ("", "flow ACCEPT OK"),
                ("", "session opened"),
            ]
        )

        self.assertEqual(
            verdict_cells(frame).tolist(),
            ["missing/block", "present/block", "missing/allow", "missing/none"],
        )

    def test_fit_returns_majority_label_of_pure_cell(self):
        frame = _frame([(None, "Deny tcp")] * 5 + [("Cisco", "Deny tcp")] * 3)
        labels = pd.Series(["malicious"] * 5 + ["suspicious"] * 3)

        rule = fit_verdict_rule(frame, labels, min_rows=5, min_purity=1.0)

        self.assertEqual(rule["cell"], "missing/block")
        self.assertEqual(rule["label"], "malicious")
        self.assertEqual(rule["support"], 5)

    def test_fit_rejects_impure_or_unsupported_cell(self):
        frame = _frame([(None, "Deny tcp")] * 4)
        with self.assertRaisesRegex(ValueError, "not pure"):
            fit_verdict_rule(frame, pd.Series(["malicious"] * 3 + ["benign"]), min_rows=1, min_purity=0.9)
        with self.assertRaisesRegex(ValueError, "not supported"):
            fit_verdict_rule(frame, pd.Series(["malicious"] * 4), min_rows=5)

    def test_apply_only_changes_rows_in_the_cell(self):
        frame = _frame([(None, "TRAFFIC,drop,udp"), ("Palo Alto", "TRAFFIC,drop,udp"), (None, "login ok")])
        rule = {"cell": "missing/block", "label": "malicious"}

        result = apply_verdict_rule(frame, ["benign", "suspicious", "benign"], rule)

        self.assertEqual(result.tolist(), ["malicious", "suspicious", "benign"])


if __name__ == "__main__":
    unittest.main()
