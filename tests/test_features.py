import unittest

import pandas as pd

from soc_baseline.features import (
    build_log_documents,
    firewall_action,
    normalize_message,
    normalize_message_timefree,
    port_bucket,
)


class FeatureTests(unittest.TestCase):
    def test_port_bucket_handles_missing_invalid_and_ranges(self):
        self.assertEqual(port_bucket(None), "missing")
        self.assertEqual(port_bucket(""), "missing")
        self.assertEqual(port_bucket("abc"), "invalid")
        self.assertEqual(port_bucket("443"), "well_known")
        self.assertEqual(port_bucket("8080"), "registered")
        self.assertEqual(port_bucket("55000"), "ephemeral")

    def test_build_log_documents_includes_structured_tokens_and_message(self):
        df = pd.DataFrame(
            {
                "timestamp": [1_722_429_000.0],
                "pipeline": ["aws_cloudtrail"],
                "src_ip": ["10.100.3.65"],
                "dst_ip": [""],
                "src_port": ["443"],
                "src_host": ["HOST-500049"],
                "dst_host": [None],
                "username": ["USER-503713"],
                "message_sanitized": ["ConsoleLogin failure from 10.100.3.65"],
                "product_name": ["AWS Instance Backup"],
                "vendor_name": ["Amazon Web Services"],
            }
        )

        doc = build_log_documents(df).iloc[0]

        self.assertIn("pipeline=aws_cloudtrail", doc)
        self.assertIn("src_ip=10.100.3.65", doc)
        self.assertIn("src_ip_prefix=10.100.3", doc)
        self.assertIn("dst_ip=missing", doc)
        self.assertIn("src_port_bucket=well_known", doc)
        self.assertIn("src_host=HOST-500049", doc)
        self.assertIn("dst_host=missing", doc)
        self.assertIn("product_name=AWS_Instance_Backup", doc)
        self.assertIn("vendor_name=Amazon_Web_Services", doc)
        self.assertIn("ConsoleLogin failure", doc)

    def _asa_frame(self, timestamp):
        return pd.DataFrame(
            {
                "timestamp": [timestamp],
                "pipeline": ["syslog"],
                "src_ip": ["100.64.3.240"],
                "dst_ip": ["172.23.186.144"],
                "src_port": ["43543"],
                "src_host": ["172.31.221.16"],
                "dst_host": [None],
                "username": ["USER-503713"],
                "message_sanitized": [
                    "<180>Dec 30 USER-9564 22:45:31: USER-0010-0324 Deny tcp src "
                    "outside:100.64.64.199/43543 dst DMZ:172.29.159.245/23 by ORG-1738-group"
                ],
                "product_name": [None],
                "vendor_name": [None],
            }
        )

    def test_content_feature_set_drops_time_and_identifier_tokens(self):
        doc = build_log_documents(self._asa_frame(1_655_105_535.0), "content").iloc[0]

        self.assertNotIn("timestamp_", doc)
        self.assertNotIn("_prefix=", doc)
        self.assertNotIn("100.64", doc)
        self.assertNotIn("USER-", doc)
        self.assertNotIn("Dec 30", doc)
        self.assertIn("pipeline=syslog", doc)
        self.assertIn("vendor_name=missing", doc)
        self.assertIn("src_ip_kind=ipv4", doc)
        self.assertIn("dst_host_kind=missing", doc)
        self.assertIn("username_kind=other", doc)
        self.assertIn("src_port_bucket=registered", doc)
        self.assertIn("message_empty=no", doc)
        self.assertIn("Deny tcp src outside:tok_ip/tok_num dst DMZ:tok_ip/23", doc)

    def test_content_documents_do_not_depend_on_timestamp(self):
        early = build_log_documents(self._asa_frame(1_655_105_535.0), "content")
        late = build_log_documents(self._asa_frame(1_722_429_000.0), "content")
        self.assertEqual(early.iloc[0], late.iloc[0])

    def test_unknown_feature_set_raises(self):
        with self.assertRaises(ValueError):
            build_log_documents(self._asa_frame(1.0), "nope")

    def test_normalize_message_replaces_dates_ids_and_numbers(self):
        normalized = normalize_message(
            pd.Series(
                [
                    "<134>1 USER-9546-07-26T06:28:52.711895-05:00 ORG-3043 started",
                    "Jan  6 12:25:01 sshd[43386]: disconnect from 10.182.42.183 Monitor",
                    "eni-083fb4fb2d7048814 1722336259 2024-07-26 Sunday",
                    None,
                ]
            )
        ).tolist()

        self.assertEqual(normalized[0], "<134>1 tok_userTtok_time tok_org started")
        self.assertEqual(normalized[1], "tok_date tok_time sshd[tok_num]: disconnect from tok_ip Monitor")
        self.assertEqual(normalized[2], "eni-tok_hex tok_epoch tok_date tok_weekday")
        self.assertEqual(normalized[3], "")

    def test_timefree_normalizer_leaves_no_digit_or_calendar_word(self):
        normalized = normalize_message_timefree(
            pd.Series(
                [
                    "<14>May  8 00:04:50 USER-0010-0051 1,USER-9564/05/08 00:04:49,TRAFFIC,drop,udp",
                    "2 100000000001 eni-0ec1cf3543aa16eec 10.100.6.190 167CRED-25166941 REJECT OK",
                    "2 100000013063 ORG-1504 100.64.0.230 17CRED-CRED-28950023 REJECT OK",
                    "Sunday 11:28 PM EDT",
                ]
            )
        ).tolist()

        self.assertEqual(normalized[0], "<0>tok_cal  0 0:0:0 0 0,0/0/0 0:0:0,TRAFFIC,drop,udp")
        self.assertEqual(normalized[1], "0 0 eni-tok_hex 0.0.0.0 0 REJECT OK")
        self.assertEqual(normalized[2], "0 0 0 0.0.0.0 0 REJECT OK")
        self.assertEqual(normalized[3], "tok_cal 0:0 tok_cal tok_cal")

    def test_timefree_documents_do_not_depend_on_any_date(self):
        frame_2022 = self._asa_frame(1_662_097_857.0)
        frame_2022.loc[0, "message_sanitized"] = (
            "<164>Sep 02 2022 05:50:57: USER-0010-0324 Deny tcp src outside:100.64.134.154/58534 "
            "dst DMZ:10.76.132.170/1962 by ORG-1738-group"
        )
        frame_2024 = self._asa_frame(1_721_993_000.0)
        frame_2024.loc[0, "message_sanitized"] = (
            "<180>Jul 28 USER-9564 13:36:43: USER-0010-0324 Deny tcp src outside:100.64.82.105/49012 "
            "dst DMZ:172.17.226.234/CRED-24615 by ORG-1738-group"
        )

        doc_2022 = build_log_documents(frame_2022, "timefree").iloc[0]
        doc_2024 = build_log_documents(frame_2024, "timefree").iloc[0]

        self.assertEqual(doc_2022, doc_2024)
        self.assertNotRegex(doc_2022.split("message_empty=no ", 1)[1], r"[1-9]")

    def test_firewall_action_prefers_block_and_ignores_substrings(self):
        actions = firewall_action(
            pd.Series(
                [
                    "TRAFFIC,drop,udp",
                    "decision=blocked accepted",
                    "flow ACCEPT OK",
                    "AccessDenied by policy",
                    '{"X-Xss-Protection":["1; mode=block"],"status":"allowed"}',
                    '{\\"X-Frame-Options\\":\\"DENY\\"}',
                    "CEF:0|WAF|act=DENY msg=x",
                    "fqdn=HOST-500127 ::: HOST-0121=BLOCKED",
                    None,
                ]
            )
        ).tolist()

        self.assertEqual(actions, ["block", "block", "allow", "none", "none", "none", "block", "block", "none"])

    def test_timefree_documents_tag_action_with_vendor_presence(self):
        frame = self._asa_frame(1.0)
        vendorless = build_log_documents(frame, "timefree").iloc[0]
        frame.loc[0, "vendor_name"] = "Cisco"
        with_vendor = build_log_documents(frame, "timefree").iloc[0]

        self.assertIn("fw_action=block", vendorless)
        self.assertIn("fw_action_vendor=block_missing", vendorless)
        self.assertIn("fw_action_vendor=block_present", with_vendor)


if __name__ == "__main__":
    unittest.main()

