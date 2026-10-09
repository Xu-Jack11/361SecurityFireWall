"""Classifier + LLM hybrid for SOC event classification (I16).

Built from train.parquet alone: no model, mask, rule or prediction from the
earlier soc_baseline iterations is used. A leakage-safe text classifier labels
every event; events it is unsure about go to a local LLM, which judges the
event from its content; a benign-first decision rule keeps benign events from
being raised as alerts.
"""
