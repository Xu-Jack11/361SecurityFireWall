"""Zero-shot reasoning probe: can a local LLM label log formats it was never shown?

The TF-IDF models only know the source formats of train.parquet. Where a
format is new (Palo Alto TRAFFIC, Symantec DLP, Microsoft Graph, Crowdstrike
asset records) they fail or need a hand-written rule (I13). This probe asks an
instruction-tuned LLM to reason about one event at a time and pick a label
from written definitions only -- no training rows, no few-shot examples -- so
every format is unseen to it, and it measures how far the model's own reading
of a log line agrees with the competition labels.

The model sees the structured fields and the message with every date, time,
epoch and year masked. IPs, ports and other numbers stay: they carry meaning
(port 445, private vs public ranges) and, with no labeled examples in the
prompt, there is no date -> label shortcut for the model to pick up.

Sample (``sample``): train rows from every (source, label) cell, plus valid
rows from the formats the TF-IDF models never saw or saw too little of. Each
cell takes one row per time-free message template first, then fills up with
other distinct messages (ports and addresses differ, the template does not).
valid answers are used only to choose and score rows, never to change the prompt.

Routing (``route``): a full valid submission that keeps G (time-free model +
source mask + verdict rule) and hands only the rows of sources G's source mask
leaves unrestricted -- fewer than 100 train rows, i.e. rare or unseen formats --
to the LLM. The routing rule and the label mapping come from train alone.

Usage (run with .venv/bin/python):
  python scripts/llm_reasoning_probe.py sample
  python scripts/llm_reasoning_probe.py run [--model Qwen/Qwen3.5-4B] [--no-thinking] [--limit 40]
  python scripts/llm_reasoning_probe.py score [--model Qwen/Qwen3.5-4B] [--no-thinking]
  python scripts/llm_reasoning_probe.py route [--model Qwen/Qwen3.5-4B] [--no-thinking]
--model takes a hub id or a local directory (e.g. a ModelScope download).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from soc_baseline.constants import ID_COLUMN, LABEL_COLUMN, LABELS, PRED_COLUMN
from soc_baseline.features import firewall_action, normalize_message_timefree
from soc_baseline.source_mask import source_keys
from soc_baseline.submission import validate_submission_frame

OUT = Path("artifacts/llm_probe")
SAMPLE_PATH = OUT / "sample.parquet"
RANDOM_STATE = 42
FIELDS = ("pipeline", "vendor_name", "product_name", "src_ip", "dst_ip", "src_port", "src_host", "dst_host", "username")
MAX_MESSAGE_CHARS = 3000
ROUTE_BASE = Path("artifacts/timefree/res_verdict_rule.csv")
ROUTE_MASK = Path("artifacts/timefree/source_label_mask.json")

# Rows per train (source, label) cell; capped by the number of distinct templates.
TRAIN_CELL_ROWS = {"benign": 12, "suspicious": 40, "malicious": 120}
TRAIN_VENDORLESS_BENIGN_ROWS = 40
# valid cells: (group, row filter, rows). Formats the TF-IDF models never saw or
# saw only as another label, plus the vendor-less rows that hold all malicious.
PAN_TRAFFIC = ",TRAFFIC,"
VALID_CELLS = (
    ("valid/malicious/palo_alto_traffic", lambda f: f["message_sanitized"].str.contains(PAN_TRAFFIC, regex=False, na=False) & (f[LABEL_COLUMN] == "malicious"), 60),
    ("valid/malicious/vendorless_other", lambda f: ~f["message_sanitized"].str.contains(PAN_TRAFFIC, regex=False, na=False) & (f[LABEL_COLUMN] == "malicious"), 60),
    ("valid/benign/vendorless", lambda f: (f["source"] == "syslog||") & (f[LABEL_COLUMN] == "benign"), 40),
    ("valid/benign/crowdstrike", lambda f: (f["vendor_name"] == "Crowdstrike") & (f[LABEL_COLUMN] == "benign"), 60),
    ("valid/suspicious/crowdstrike", lambda f: (f["vendor_name"] == "Crowdstrike") & (f[LABEL_COLUMN] == "suspicious"), 60),
    ("valid/benign/microsoft_graph", lambda f: f["product_name"] == "Graph", 30),
    ("valid/suspicious/symantec_dlp", lambda f: f["product_name"] == "Symantec Data Loss Prevention", 9),
    ("valid/benign/duo", lambda f: (f["product_name"] == "Duo") & (f[LABEL_COLUMN] == "benign"), 20),
    ("valid/suspicious/duo", lambda f: (f["product_name"] == "Duo") & (f[LABEL_COLUMN] == "suspicious"), 1),
)

_MONTHS = (
    "January|February|March|April|June|July|August|September|October|November|December|"
    "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec"
)
_WEEKDAYS = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun"
# Applied in order. The sanitizer rewrote some years into IDs ("USER-9564/03/11",
# "USER-9564-03-11T12:32:40"), so those dates go first. Times must not eat
# "10.0.0.1:443" or the next CSV field ("12:32:40,10.100.4.198"), hence the
# look-arounds and the 3-6 digit fraction.
_TIME_MASKS = (
    (re.compile(r"\b(?:USER|ORG|CRED|HOST)-\d+[-/]\d{1,2}[-/]\d{1,2}"), "<DATE>"),
    (re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}"), "<DATE>"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "<DATE>"),
    (re.compile(rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:,?\s+\d{{4}})?\b"), "<DATE>"),
    (re.compile(rf"\b\d{{1,2}}[\s-](?:{_MONTHS})[\s-]\d{{2,4}}\b"), "<DATE>"),
    (re.compile(rf"\b(?:{_WEEKDAYS})\b"), "<DAY>"),
    (re.compile(r"(?<![\d.:])\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{3,6}(?![\d.]))?(?:\s?[AP]M)?(?:Z|\s?[+-]\d{2}:?\d{2})?(?!\d)"), "<TIME>"),
    (re.compile(r"\b1[4-8]\d{8}(?:\d{3})?(?:\.\d+)?\b"), "<EPOCH>"),
    (re.compile(r"\b20[12]\d\b"), "<YEAR>"),
)

SYSTEM_PROMPT = """You are a SOC analyst triaging security log events one at a time.
Classify the event into exactly one label:
- benign: routine operational activity with no security concern, e.g. normal logons, administrative or configuration changes, backups, asset or inventory records, allowed traffic, health and error messages.
- suspicious: a security control flagged or blocked something, or the event otherwise deserves analyst review, but nothing shows an attack succeeded, e.g. firewall or ACL denies and drops, rejected connections, IDS/EDR/DLP detections, MFA denials, WAF blocks, repeated failed logons.
- malicious: evidence of actual hostile activity, e.g. exploitation, malware execution, command-and-control, unauthorized access that succeeded, data exfiltration, or attack tooling.

Notes on the data:
- The logs are anonymized. Tokens such as USER-0123, HOST-4567, CRED-89 or ORG-0152 are placeholders that may replace any substring (names, words, numbers), so text can look garbled.
- Dates and times are masked as <DATE>, <TIME>, <EPOCH>, <YEAR>, <DAY>.
- Fields may be empty; an empty vendor or product only means that metadata is missing.

Reason briefly about what the event records and what the reporting device did, then end your reply with one line of JSON:
{"label": "benign|suspicious|malicious", "verdict": "block|allow|none", "event_type": "<a few words>", "confidence": <0 to 1>}
where verdict is what the device did with the activity (none if the event is not an access decision)."""

_JSON_RE = re.compile(r"\{[^{}]*\"label\"[^{}]*\}", re.DOTALL)
_LABEL_WORD_RE = re.compile(r"\b(benign|suspicious|malicious)\b", re.IGNORECASE)


def mask_time(text: str) -> str:
    for pattern, replacement in _TIME_MASKS:
        text = pattern.sub(replacement, text)
    return text


def message_view(text: str) -> str:
    text = mask_time(text.strip())
    if len(text) > MAX_MESSAGE_CHARS:
        text = text[:MAX_MESSAGE_CHARS] + " ...[truncated]"
    return text


def user_prompt(row: pd.Series) -> str:
    lines = [f"{field}: {row[field] if str(row[field]).strip() else '(empty)'}" for field in FIELDS]
    return "\n".join(lines) + "\nmessage:\n" + row["message_view"]


def _with_source(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for field in FIELDS + ("message_sanitized",):
        frame[field] = frame[field].fillna("").astype(str)
    frame["source"] = frame["pipeline"] + "|" + frame["vendor_name"] + "|" + frame["product_name"]
    return frame


def _take_rows(cell: pd.DataFrame, rows: int) -> pd.DataFrame:
    """Up to ``rows`` distinct messages of ``cell``: one per time-free template first, then the rest."""

    pool = cell.sample(n=min(len(cell), rows * 50), random_state=RANDOM_STATE)
    pool = pool.loc[~pool["message_sanitized"].duplicated()]
    first = ~normalize_message_timefree(pool["source"] + " " + pool["message_sanitized"]).duplicated()
    return pd.concat([pool[first], pool[~first]]).head(rows)


def build_sample() -> pd.DataFrame:
    columns = [ID_COLUMN, *FIELDS, "message_sanitized"]
    train = _with_source(pd.read_parquet("data/train.parquet", columns=columns + [LABEL_COLUMN]))
    parts = []
    for (source, label), cell in train.groupby(["source", LABEL_COLUMN]):
        rows = TRAIN_VENDORLESS_BENIGN_ROWS if (source == "syslog||" and label == "benign") else TRAIN_CELL_ROWS[label]
        parts.append(_take_rows(cell, rows).assign(split="train", group=f"train/{label}/{source}"))
    del train

    valid = pd.read_parquet("data/valid_input.parquet", columns=columns)
    valid = _with_source(valid.merge(pd.read_parquet("data/valid_answer_private.parquet"), on=ID_COLUMN))
    for group, keep, rows in VALID_CELLS:
        parts.append(_take_rows(valid[keep(valid)], rows).assign(split="valid", group=group))

    sample = pd.concat(parts, ignore_index=True)
    sample["message_view"] = sample["message_sanitized"].map(message_view)
    sample["regex_verdict"] = firewall_action(sample["message_sanitized"]).to_numpy()
    return sample.drop(columns=["message_sanitized"])


def run(
    model: str,
    frame: pd.DataFrame,
    outputs_path: Path,
    limit: int | None,
    max_tokens: int,
    chunk: int,
    max_num_seqs: int,
    thinking: bool,
) -> None:
    """Generate a reply for every row of ``frame`` not yet in ``outputs_path`` (appended per chunk)."""

    from vllm import LLM, SamplingParams

    done = set()
    if outputs_path.exists():
        done = {json.loads(line)[ID_COLUMN] for line in outputs_path.read_text(encoding="utf-8").splitlines()}
    todo = frame[~frame[ID_COLUMN].isin(done)]
    if limit is not None and "group" in todo:
        # A small run should still touch every group: take the groups round-robin.
        todo = todo.iloc[np.argsort(todo.groupby("group").cumcount().to_numpy(), kind="stable")].head(limit)
    print(f"{len(done)} rows already done, {len(todo)} to run")
    if todo.empty:
        return

    # Sized for a 15 GB T4 next to 8 GB of weights. Qwen3.5 keeps a recurrent
    # state per sequence in its linear-attention layers, so vLLM's default of 256
    # concurrent sequences does not fit. The linear-attention prefill kernel needs
    # scratch memory that vLLM's profiling (one long dummy sequence) misses when
    # many short prompts prefill together, hence small chunks and 20% headroom.
    llm = LLM(model=model, dtype="float16", max_model_len=10240, gpu_memory_utilization=0.80,
              max_num_seqs=max_num_seqs, max_num_batched_tokens=1024,
              limit_mm_per_prompt={"image": 0, "video": 0}, seed=RANDOM_STATE)
    # Qwen3.5 model card: thinking mode / instruct mode for general tasks. Without
    # thinking the model still writes the brief reasoning the prompt asks for.
    sampling = dict(temperature=1.0, top_p=0.95) if thinking else dict(temperature=0.7, top_p=0.8)
    params = SamplingParams(**sampling, top_k=20, min_p=0.0, presence_penalty=1.5, max_tokens=max_tokens, seed=RANDOM_STATE)
    with outputs_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(todo), chunk):
            batch = todo.iloc[start : start + chunk]
            conversations = [
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt(row)}]
                for _, row in batch.iterrows()
            ]
            began = time.time()
            results = llm.chat(conversations, params, use_tqdm=False, chat_template_kwargs={"enable_thinking": thinking})
            seconds = time.time() - began
            generated = 0
            for (_, row), result in zip(batch.iterrows(), results):
                completion = result.outputs[0]
                generated += len(completion.token_ids)
                record = {
                    ID_COLUMN: row[ID_COLUMN],
                    "prompt_tokens": len(result.prompt_token_ids),
                    "output_tokens": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                    "thinking": thinking,
                    "text": completion.text,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"rows {start + len(batch)}/{len(todo)}: {seconds:.0f}s, {generated / seconds:.0f} output tok/s", flush=True)


def read_replies(outputs_path: Path) -> pd.DataFrame:
    """One parsed reply per event_id (the last one written wins)."""

    records = [json.loads(line) for line in outputs_path.read_text(encoding="utf-8").splitlines()]
    replies = pd.DataFrame(records).drop_duplicates(ID_COLUMN, keep="last")
    thinking = replies["thinking"] if "thinking" in replies else pd.Series(True, index=replies.index)
    parsed = pd.DataFrame([parse_answer(text, bool(t)) for text, t in zip(replies["text"], thinking)], index=replies.index)
    return pd.concat([replies.drop(columns=["text"]), parsed], axis=1)


def route_frame() -> pd.DataFrame:
    """valid rows whose source G's source mask leaves unrestricted (< 100 train rows)."""

    supported = set(json.loads(ROUTE_MASK.read_text(encoding="utf-8")))
    valid = pd.read_parquet("data/valid_input.parquet", columns=[ID_COLUMN, *FIELDS, "message_sanitized"])
    routed = _with_source(valid[~source_keys(valid).isin(supported).to_numpy()])
    routed["message_view"] = routed["message_sanitized"].map(message_view)
    return routed.drop(columns=["message_sanitized"])


def write_routed_submission(routed: pd.DataFrame, outputs_path: Path, path: Path) -> None:
    """G's submission with the routed rows replaced by the LLM's label.

    The LLM's "malicious" becomes "suspicious": in the train probe no record it
    called malicious was labeled malicious (they were Crowdstrike detections,
    labeled suspicious), and the competition's malicious rows are vendor-less
    records that G's verdict rule already covers. Rows without a parsed label
    keep G's prediction.
    """

    base = pd.read_csv(ROUTE_BASE, dtype={ID_COLUMN: str})
    replies = read_replies(outputs_path).set_index(ID_COLUMN)["label"].dropna()
    llm = replies.reindex(routed[ID_COLUMN]).dropna().replace({"malicious": "suspicious"})
    submission = base.set_index(ID_COLUMN)
    changed = (submission.loc[llm.index, PRED_COLUMN] != llm).sum()
    submission.loc[llm.index, PRED_COLUMN] = llm
    submission = submission.reset_index()
    validate_submission_frame(submission, base[ID_COLUMN])
    submission.to_csv(path, index=False)
    print(routed["source"].value_counts().to_string())
    print(f"{len(routed)} routed rows, {len(llm)} with an LLM label, {changed} changed vs G -> {path}")
    print("LLM labels on routed rows:", llm.value_counts().to_dict())


def parse_answer(text: str, thinking: bool = True) -> dict:
    """Label and attributes from the reply (after </think> when thinking); label None if there is none."""

    answer = text
    if thinking:
        _, closed, answer = text.partition("</think>")
        if not closed:
            return {"label": None, "parse": "no_answer"}
    for candidate in reversed(_JSON_RE.findall(answer)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        label = str(parsed.get("label", "")).strip().lower()
        if label in LABELS:
            return {**parsed, "label": label, "parse": "json"}
    words = _LABEL_WORD_RE.findall(answer)
    return {"label": words[-1].lower() if words else None, "parse": "word" if words else "none"}


def _rates(truth: pd.Series, pred: pd.Series) -> dict:
    alert_truth, alert_pred = truth != "benign", pred.isin(["suspicious", "malicious"])
    return {
        "rows": int(len(truth)),
        "accuracy": round(float((truth == pred).mean()), 4),
        "alert_accuracy": round(float((alert_truth == alert_pred).mean()), 4),
    }


def score(out: Path) -> None:
    sample = pd.read_parquet(SAMPLE_PATH)
    frame = sample.merge(read_replies(out / "outputs.jsonl"), on=ID_COLUMN, how="inner")
    frame["pred"] = frame["label"].fillna("unparsed")
    frame["llm_verdict"] = frame.get("verdict", pd.Series(index=frame.index, dtype=object)).fillna("missing")

    # The TF-IDF versions on the same valid rows (they trained on the train rows).
    tfidf = {}
    for name, path in {"F_timefree_mask": "artifacts/timefree/res_timefree.csv", "G_verdict_rule": "artifacts/timefree/res_verdict_rule.csv"}.items():
        predictions = pd.read_csv(path)
        tfidf[name] = frame[[ID_COLUMN]].merge(predictions, on=ID_COLUMN, how="left")["pred_label"].to_numpy()

    results: dict = {
        "rows": int(len(frame)),
        "parse": frame["parse"].value_counts().to_dict(),
        "finish_reason": frame["finish_reason"].value_counts().to_dict(),
        "output_tokens": frame["output_tokens"].describe(percentiles=[0.5, 0.9, 0.99]).round(0).to_dict(),
        "groups": {},
        "splits": {},
    }
    for group, rows in frame.groupby("group", sort=True):
        entry = {**_rates(rows[LABEL_COLUMN], rows["pred"]), "pred": rows["pred"].value_counts().to_dict()}
        if rows["split"].iat[0] == "valid":
            for name, predictions in tfidf.items():
                entry[name] = _rates(rows[LABEL_COLUMN], pd.Series(predictions[rows.index], index=rows.index))
        results["groups"][group] = entry
    for split, rows in frame.groupby("split"):
        entry = {
            **_rates(rows[LABEL_COLUMN], rows["pred"]),
            "macro_f1": round(float(f1_score(rows[LABEL_COLUMN], rows["pred"], labels=list(LABELS), average="macro", zero_division=0)), 4),
            "confusion_matrix(rows=truth,cols=benign/suspicious/malicious/unparsed)": confusion_matrix(
                rows[LABEL_COLUMN], rows["pred"], labels=[*LABELS, "unparsed"]
            )[:3].tolist(),
        }
        if split == "valid":
            for name, predictions in tfidf.items():
                entry[name] = _rates(rows[LABEL_COLUMN], pd.Series(predictions[rows.index], index=rows.index))
        results["splits"][split] = entry
    results["verdict_vs_regex"] = pd.crosstab(frame["regex_verdict"], frame["llm_verdict"]).to_dict(orient="index")

    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    keep = [ID_COLUMN, "split", "group", LABEL_COLUMN, "pred", "llm_verdict", "regex_verdict", "event_type", "confidence", "parse", "output_tokens"]
    frame.reindex(columns=keep).to_csv(out / "predictions.csv", index=False)
    print(json.dumps({key: results[key] for key in ("rows", "parse", "finish_reason", "splits")}, indent=2))
    for group, entry in results["groups"].items():
        extra = "  ".join(f"{name} {entry[name]['accuracy']:.2f}" for name in tfidf if name in entry)
        print(f"{group:<70} n={entry['rows']:>3}  acc {entry['accuracy']:.2f}  alert_acc {entry['alert_accuracy']:.2f}  {extra}  {entry['pred']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["sample", "run", "score", "route"])
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--limit", type=int, default=None, help="run: only this many new rows, spread over groups")
    parser.add_argument("--no-thinking", action="store_true", help="instruct mode; results go to <model>-nothink")
    parser.add_argument("--max-tokens", type=int, default=None, help="run: default 6144 with thinking, 1024 without")
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=32, help="run: concurrent sequences in vLLM")
    args = parser.parse_args()
    out = OUT / (args.model.rstrip("/").split("/")[-1] + ("-nothink" if args.no_thinking else ""))
    out.mkdir(parents=True, exist_ok=True)

    if args.command == "sample":
        sample = build_sample()
        sample.to_parquet(SAMPLE_PATH, index=False)
        print(sample.groupby(["split", "group"]).size().to_string())
        print(f"{len(sample)} rows -> {SAMPLE_PATH}")
    elif args.command == "score":
        score(out)
    else:
        max_tokens = args.max_tokens or (1024 if args.no_thinking else 6144)
        if args.command == "run":
            run(args.model, pd.read_parquet(SAMPLE_PATH), out / "outputs.jsonl", args.limit, max_tokens,
                args.chunk, args.max_num_seqs, not args.no_thinking)
        else:
            routed = route_frame()
            run(args.model, routed, out / "route_outputs.jsonl", None, max_tokens, args.chunk, args.max_num_seqs,
                not args.no_thinking)
            write_routed_submission(routed, out / "route_outputs.jsonl", out / "res_route.csv")


if __name__ == "__main__":
    main()
