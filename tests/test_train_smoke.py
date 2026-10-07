import tempfile
import unittest
from pathlib import Path

import pandas as pd

from soc_baseline.train import BaselineConfig, run_baseline


class TrainSmokeTests(unittest.TestCase):
    def test_run_baseline_trains_predicts_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.parquet"
            test_path = tmp_path / "valid_input.parquet"
            output_path = tmp_path / "res.csv"
            artifacts_dir = tmp_path / "artifacts"

            train = pd.DataFrame(
                {
                    "event_id": [f"tr{i}" for i in range(12)],
                    "timestamp": [1_722_429_000.0 + i for i in range(12)],
                    "pipeline": ["syslog"] * 4 + ["aws_cloudtrail"] * 4 + ["network_flows"] * 4,
                    "src_ip": ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"] * 3,
                    "dst_ip": ["10.0.1.1", "", "", "10.0.1.4"] * 3,
                    "src_port": ["443", "55555", "", "8080"] * 3,
                    "src_host": ["HOST-A", "HOST-B", "HOST-C", "HOST-D"] * 3,
                    "dst_host": ["HOST-X", "", "", "HOST-Y"] * 3,
                    "username": ["USER-A", "USER-B", "", "USER-D"] * 3,
                    "message_sanitized": [
                        "normal backup completed",
                        "normal login success",
                        "malware beacon detected",
                        "suspicious failed login",
                    ]
                    * 3,
                    "product_name": ["Windows Logs"] * 12,
                    "vendor_name": ["Microsoft"] * 12,
                    "label_binary": ["benign", "benign", "malicious", "suspicious"] * 3,
                }
            )
            test = train.drop(columns=["label_binary"]).iloc[:4].copy()
            test["event_id"] = [f"te{i}" for i in range(4)]
            train.to_parquet(train_path)
            test.to_parquet(test_path)

            result = run_baseline(
                BaselineConfig(
                    train_path=train_path,
                    test_path=test_path,
                    output_path=output_path,
                    artifacts_dir=artifacts_dir,
                    max_train_rows=None,
                    max_test_rows=None,
                    test_size=0.25,
                    max_features=200,
                    min_df=1,
                    top_features=5,
                    prediction_chunk_size=2,
                    random_state=3,
                )
            )

            submission = pd.read_csv(output_path)
            self.assertEqual(len(submission), 4)
            self.assertEqual(set(submission["event_id"]), set(test["event_id"]))
            self.assertTrue(set(submission["pred_label"]).issubset({"benign", "malicious", "suspicious"}))
            self.assertTrue((artifacts_dir / "metrics.json").exists())
            self.assertTrue((artifacts_dir / "classification_report.csv").exists())
            self.assertTrue((artifacts_dir / "feature_importance.csv").exists())
            self.assertEqual(result["submission_rows"], 4)

    def test_run_baseline_with_content_features_and_source_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train = pd.DataFrame(
                {
                    "event_id": [f"tr{i}" for i in range(12)],
                    "timestamp": [1_722_429_000.0 + i for i in range(12)],
                    "pipeline": ["syslog"] * 12,
                    "src_ip": ["10.0.0.1"] * 12,
                    "dst_ip": [""] * 12,
                    "src_port": ["443"] * 12,
                    "src_host": ["HOST-A"] * 12,
                    "dst_host": [""] * 12,
                    "username": [""] * 12,
                    "message_sanitized": ["backup completed", "malware beacon", "failed login"] * 4,
                    "product_name": ["Backup", "", "Firewall"] * 4,
                    "vendor_name": ["AWS", "", "Cisco"] * 4,
                    "label_binary": ["benign", "malicious", "suspicious"] * 4,
                }
            )
            # The AWS/Backup source only ever produced benign rows, so the mask
            # must keep a malware-looking AWS/Backup row benign.
            test = pd.DataFrame(
                {
                    "event_id": ["te0", "te1"],
                    "timestamp": [1_600_000_000.0, 1_600_000_001.0],
                    "pipeline": ["syslog", "syslog"],
                    "src_ip": ["10.9.9.9", "10.9.9.9"],
                    "dst_ip": ["", ""],
                    "src_port": ["443", "443"],
                    "src_host": ["HOST-Z", "HOST-Z"],
                    "dst_host": ["", ""],
                    "username": ["", ""],
                    "message_sanitized": ["malware beacon", "malware beacon"],
                    "product_name": ["Backup", ""],
                    "vendor_name": ["AWS", ""],
                }
            )
            train.to_parquet(tmp_path / "train.parquet")
            test.to_parquet(tmp_path / "valid_input.parquet")
            artifacts_dir = tmp_path / "artifacts"

            result = run_baseline(
                BaselineConfig(
                    train_path=tmp_path / "train.parquet",
                    test_path=tmp_path / "valid_input.parquet",
                    output_path=tmp_path / "res.csv",
                    artifacts_dir=artifacts_dir,
                    max_train_rows=None,
                    test_size=0.25,
                    max_features=200,
                    min_df=1,
                    top_features=5,
                    random_state=3,
                    model_backend="sklearn",
                    feature_set="content",
                    source_label_mask=True,
                    source_mask_min_rows=2,
                    benign_weight=10.0,
                )
            )

            submission = pd.read_csv(tmp_path / "res.csv").set_index("event_id")["pred_label"]
            self.assertEqual(submission["te0"], "benign")
            self.assertEqual(result["feature_set"], "content")
            self.assertTrue(result["source_label_mask"])
            self.assertEqual(result["benign_weight"], 10.0)
            self.assertTrue((artifacts_dir / "source_label_mask.json").exists())


if __name__ == "__main__":
    unittest.main()

