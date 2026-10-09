import unittest

import numpy as np
import pandas as pd

from soc_hybrid.decision import benign_first, cost_confidence, cost_matrix, decision_confidence, fuse_codes, min_cost
from soc_hybrid.llm import parse_reply, system_prompt
from soc_hybrid.metrics import evaluate
from soc_hybrid.models import training_weights
from soc_hybrid.route_select import fuse
from soc_hybrid.text import classifier_documents, mask_for_llm, normalize_for_classifier


class TextTest(unittest.TestCase):
    def test_classifier_view_drops_every_time_value_and_pseudonym(self):
        a = '<180>Sep 01 USER-9564 00:01:56: USER-0010-0324 Deny tcp src outside:100.64.109.240/59237'
        b = '<164>Jul 26 USER-9546 06:41:20: USER-0147-0001 Deny tcp src outside:100.64.1.2/22'
        self.assertEqual(normalize_for_classifier(a), normalize_for_classifier(b))
        self.assertNotIn("9564", normalize_for_classifier(a))
        self.assertIn("zuser", normalize_for_classifier(a))

    def test_calendar_words_glued_to_digits_are_masked_but_words_are_not(self):
        self.assertIn("zmon", normalize_for_classifier("S Jul20 0:00 [kdmflush]"))
        self.assertIn("market", normalize_for_classifier("Market").lower())

    def test_llm_view_masks_time_but_keeps_ports_and_codes(self):
        masked = mask_for_llm('Jul 26 06:10:17 host "code":"4625" dst 10.0.0.1/445 at 1721992874')
        self.assertNotIn("06:10:17", masked)
        self.assertNotIn("1721992874", masked)
        self.assertIn("4625", masked)
        self.assertIn("10.0.0.1/445", masked)

    def test_documents_carry_source_and_field_shapes(self):
        frame = pd.DataFrame({"pipeline": ["syslog"], "vendor_name": ["Cisco"], "product_name": ["ASA Firewall"],
                              "src_ip": ["10.1.2.3"], "dst_ip": [""], "src_port": ["443"], "src_host": [""],
                              "dst_host": [""], "username": [""], "message_sanitized": ["Deny tcp"]})
        doc = classifier_documents(frame).iloc[0]
        self.assertIn("fvendor_cisco", doc)
        self.assertIn("fsrcip_private", doc)
        self.assertIn("fsrcport_wellknown", doc)


class DecisionTest(unittest.TestCase):
    def test_benign_first_needs_alert_w_times_as_likely(self):
        proba = np.array([[0.15, 0.85, 0.0], [0.05, 0.95, 0.0]])
        self.assertEqual(list(benign_first(proba, 1)), ["suspicious", "suspicious"])
        self.assertEqual(list(benign_first(proba, 10)), ["benign", "suspicious"])

    def test_confidence_is_half_at_a_tie(self):
        self.assertAlmostEqual(decision_confidence(np.array([[0.1, 0.9, 0.0]]), 9)[0], 0.5)

    def test_fusion_rules(self):
        proba = np.array([0.2, 0.1, 0.7])
        self.assertEqual(fuse("llm", "suspicious", "benign", proba), "suspicious")
        self.assertEqual(fuse("llm_gate", "suspicious", "benign", proba), "malicious")
        self.assertEqual(fuse("llm_gate", "benign", "malicious", proba), "benign")
        self.assertEqual(fuse("agree_alert", "suspicious", "benign", proba), "benign")
        self.assertEqual(fuse("llm", None, "benign", proba), "benign")

    def test_probability_fusion_applies_the_benign_weight_to_the_llm(self):
        proba = np.array([0.2, 0.1, 0.7])
        sure = np.array([0.05, 0.9, 0.05])
        unsure = np.array([0.2, 0.75, 0.05])
        self.assertEqual(fuse("llm_prob", "suspicious", "benign", proba, sure, 10), "suspicious")
        self.assertEqual(fuse("llm_prob", "suspicious", "benign", proba, unsure, 10), "benign")
        self.assertEqual(fuse("llm_prob_gate", "suspicious", "benign", proba, sure, 10), "malicious")

    def test_keep_malicious_fusion_never_vetoes_a_malicious_decision(self):
        proba = np.array([0.2, 0.1, 0.7])
        benign = np.array([0.9, 0.05, 0.05])
        self.assertEqual(fuse("llm_prob_gate_km", "benign", "malicious", proba, benign, 10), "malicious")
        self.assertEqual(fuse("llm_prob_gate_km", "benign", "suspicious", proba, benign, 10), "benign")


    def test_min_cost_alerts_once_a_threat_is_likelier_than_one_in_m_plus_one(self):
        proba = np.array([[0.92, 0.05, 0.03], [0.88, 0.08, 0.04]])
        self.assertEqual(list(min_cost(proba, cost_matrix(10))), ["benign", "suspicious"])
        self.assertEqual(list(min_cost(proba, cost_matrix(1))), ["benign", "benign"])

    def test_cost_confidence_is_half_at_a_tie(self):
        self.assertAlmostEqual(cost_confidence(np.array([[0.5, 0.5, 0.0]]), cost_matrix(1))[0], 0.5)
        self.assertAlmostEqual(cost_confidence(np.array([[1.0, 0.0, 0.0]]), cost_matrix(10))[0], 1.0)

    def test_cost_fusion_rules(self):
        clf = np.array([2, 1, 0])
        proba = np.array([[0.1, 0.2, 0.7], [0.3, 0.6, 0.1], [0.6, 0.1, 0.3]])
        calm = np.tile([0.95, 0.04, 0.01], (3, 1))
        alarmed = np.tile([0.3, 0.6, 0.1], (3, 1))
        # The LLM vetoes suspicious but never malicious, and raises with the classifier's alert type.
        self.assertEqual(list(fuse_codes("gate_km", 10, clf, proba, calm, 1)), [2, 0, 0])
        self.assertEqual(list(fuse_codes("gate_km", 10, clf, proba, alarmed, 1)), [2, 1, 2])
        # raise never turns an alert into benign.
        self.assertEqual(list(fuse_codes("raise", 10, clf, proba, calm, 1)), [2, 1, 0])
        self.assertEqual(list(fuse_codes("raise", 10, clf, proba, alarmed, 1)), [2, 1, 2])
        self.assertEqual(list(fuse_codes("mix_km", 0.5, clf, proba, calm, 1)), [2, 0, 0])

class RoutingTest(unittest.TestCase):
    def setUp(self):
        self.docs = pd.DataFrame({
            "source": ["A/x", "A/x", "B/y", "A/x"],
            "clf": ["benign", "malicious", "benign", "benign"],
            "conf": [0.99, 0.99, 0.99, 0.6],
            "maxsim": [0.99, 0.99, 0.95, 0.5],
        })
        self.cells = {("A/x", "benign"), ("A/x", "suspicious")}

    def test_each_trigger(self):
        from soc_hybrid.pipeline import triggers

        fired = triggers(self.docs, doubt=0.7, novelty=0.8, pair=True, unseen_source=True, train_cells=self.cells)
        self.assertEqual(list(fired["unseen_source"]), [False, False, True, False])
        self.assertEqual(list(fired["new_pair"]), [False, True, True, False])
        self.assertEqual(list(fired["novelty"]), [False, False, False, True])
        self.assertEqual(list(fired["doubt"]), [False, False, False, True])

    def test_budget_keeps_unseen_sources_then_least_similar(self):
        from soc_hybrid.pipeline import route, triggers

        fired = triggers(self.docs, doubt=0.7, novelty=0.8, pair=True, unseen_source=True, train_cells=self.cells)
        self.assertEqual(list(route(self.docs, fired, budget=2)), [False, False, True, True])

    def test_budget_zero_routes_every_triggered_document(self):
        from soc_hybrid.pipeline import route, triggers

        fired = triggers(self.docs, doubt=0.7, novelty=0.8, pair=True, unseen_source=True, train_cells=self.cells)
        self.assertEqual(list(route(self.docs, fired, budget=0)), [False, True, True, True])


class MetricsTest(unittest.TestCase):
    def test_cost_counts_benign_misses_ten_times(self):
        result = evaluate(["benign", "benign", "suspicious"], ["suspicious", "benign", "benign"])
        self.assertEqual(result["benign_misses"], 1)
        self.assertEqual(result["errors"], 2)
        self.assertEqual(result["cost"], 11)

    def test_corrected_rule_counts_threat_misses_m_times_and_swaps_at_half(self):
        result = evaluate(["benign", "suspicious", "malicious", "malicious"],
                          ["suspicious", "benign", "suspicious", "malicious"])
        self.assertEqual((result["false_alarms"], result["threat_misses"], result["swaps"]), (1, 1, 1))
        self.assertAlmostEqual(result["cost_m10"], 11.5)
        self.assertAlmostEqual(result["cost_m2"], 3.5)
        self.assertAlmostEqual(result["partial_accuracy"], 0.375)
        self.assertAlmostEqual(result["threat_recall"], 2 / 3)
        self.assertAlmostEqual(result["threat_precision"], 2 / 3)

    def test_class_balanced_weights(self):
        unique = pd.DataFrame({"label": ["benign"] * 3 + ["suspicious"], "rows": [1, 1, 1, 9]})
        weights = training_weights(unique, "doc")
        self.assertAlmostEqual(weights[:3].sum(), weights[3:].sum())


class PromptTest(unittest.TestCase):
    def test_holdout_prompt_drops_lines_read_only_off_held_out_cells(self):
        full = system_prompt("codebook")
        held_out = system_prompt("codebook", {("Crowdstrike/Falcon", "suspicious")})
        self.assertIn("EDR detection", full)
        self.assertNotIn("EDR detection", held_out)
        # A line also read off another cell stays.
        self.assertIn("Linux system logs", system_prompt("codebook", {("Linux/Linux PAM", "benign")}))
        self.assertNotIn("  - ", system_prompt("generic"))

    def test_parse_reply(self):
        self.assertEqual(parse_reply('ok {"label": "Benign", "reason": "x"}'), "benign")
        self.assertEqual(parse_reply("I think this is suspicious."), "suspicious")
        self.assertIsNone(parse_reply("no idea"))


if __name__ == "__main__":
    unittest.main()
