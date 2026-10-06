import tempfile
import unittest
from pathlib import Path

import pandas as pd

from soc_baseline.gpu_modeling import TorchTfidfClassifier, resolve_torch_device
from soc_baseline.train import BaselineConfig, run_baseline


class GpuModelingTests(unittest.TestCase):
    def test_resolve_torch_device_accepts_explicit_cpu(self):
        self.assertEqual(resolve_torch_device("cpu", require_cuda=False).type, "cpu")

    def test_torch_tfidf_classifier_trains_and_predicts_on_cpu(self):
        docs = pd.Series(
            [
                "pipeline=syslog malware beacon",
                "pipeline=syslog malware callback",
                "pipeline=aws normal backup",
                "pipeline=aws normal login",
                "pipeline=ad suspicious failure",
                "pipeline=ad suspicious password",
            ]
        )
        labels = pd.Series(["malicious", "malicious", "benign", "benign", "suspicious", "suspicious"])
        model = TorchTfidfClassifier(
            max_features=100,
            min_df=1,
            random_state=11,
            device="cpu",
            epochs=8,
            batch_size=3,
            learning_rate=0.2,
        )

        model.fit(docs, labels)
        predictions = model.predict(docs)
        probabilities = model.predict_proba(docs)

        self.assertEqual(list(model.classes_), ["benign", "suspicious", "malicious"])
        self.assertEqual(len(predictions), len(labels))
        self.assertTrue(set(predictions).issubset({"benign", "suspicious", "malicious"}))
        self.assertEqual(probabilities.shape, (6, 3))
        self.assertTrue(((probabilities.sum(axis=1) > 0.999) & (probabilities.sum(axis=1) < 1.001)).all())

    def test_run_baseline_accepts_torch_backend(self):
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
                    model_backend="torch",
                    device="cpu",
                    torch_epochs=3,
                    torch_batch_size=4,
                    torch_learning_rate=0.2,
                )
            )

            submission = pd.read_csv(output_path)
            self.assertEqual(len(submission), 4)
            self.assertEqual(result["model_backend"], "torch")
            self.assertEqual(result["device"], "cpu")
            self.assertTrue((artifacts_dir / "model.joblib").exists())


if __name__ == "__main__":
    unittest.main()

