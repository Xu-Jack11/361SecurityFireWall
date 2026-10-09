"""The LLM half: prompts built from train alone, a cached vLLM runner and reply parsing.

The label definitions are a codebook written from train's (source, label)
cells: every line names the cells it was read off. On a train-internal
holdout the lines read only off held-out cells are dropped, so the LLM is
never told the convention it is being tested on.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LABELS, OUT

MODELS = {
    "qwen3.5-4b": "/root/.cache/modelscope/hub/models/Qwen/Qwen3.5-4B",
    "qwen3-4b-2507": "/root/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507",
}
MODEL_PATH = MODELS["qwen3.5-4b"]
LLM_DIR = OUT / "llm"
MAX_MESSAGE_CHARS = 1500
PROMPT_VERSION = "v2"  # v2: replies carry the label's probability (top-5 logprobs at the label token)

# (label, description, cells it was read off). Written from train only.
CODEBOOK = (
    ("benign", "cloud backup job records, often with an empty message",
     {("Amazon Web Services/AWS Instance Backup", "benign")}),
    ("benign", "cloud VPC flow log records whose action is NODATA or SKIPDATA, and records with an empty message",
     {("/", "benign")}),
    ("benign", "Linux system logs: cron jobs, systemd sessions, sshd accepted logins and disconnects, PAM sessions opened or closed, log shipper messages",
     {("/", "benign"), ("Linux/Linux PAM", "benign")}),
    ("benign", "Windows and Active Directory audit events of normal activity: object and handle access, successful logons, the filtering platform permitting a connection, service and process events",
     {("Microsoft/Windows Active Directory", "benign"), ("Microsoft/Windows Logs", "benign"), ("SentinelOne/SentinelOne", "benign"), ("/", "benign")}),
    ("benign", "directory management and audit reports",
     {("ManageEngine/ManageEngine ADManager", "benign")}),
    ("benign", "web server and web application firewall records of ordinary requests that were allowed (HTTP 200, health checks, API calls)",
     {("Apache/Apache Web Server", "benign"), ("Barracuda/Barracuda WAF", "benign")}),
    ("benign", "network device status messages: interface link up or down, power, voltage, PoE",
     {("Cisco/Cisco Network Operating System", "benign"), ("Cisco/ASA Firewall", "benign")}),
    ("benign", "virtualization platform logs: process listings, API and proxy access",
     {("VMWare/VMWare VCenter", "benign")}),
    ("benign", "endpoint protection agent records: content updates, and traffic rule hits the policy allows or blocks as configured",
     {("Symantec/Symantec Endpoint Protection", "benign")}),
    ("benign", "email security gateway records of processed messages, including messages it filtered as spam",
     {("Barracuda/Barracuda ESS", "benign")}),
    ("benign", "MFA records of successful or remembered authentications",
     {("Cisco/Duo", "benign")}),
    ("suspicious", "firewall or ACL deny and drop records from an identified firewall, and cloud VPC flow records with action REJECT",
     {("Cisco/ASA Firewall", "suspicious"), ("Amazon Web Services/AWS VPC Security", "suspicious")}),
    ("suspicious", "failed logons (e.g. Windows 'An account failed to log on') and authentication failure alerts from security platforms",
     {("Microsoft/Windows Active Directory", "suspicious"), ("WitFoo/Precinct", "suspicious")}),
    ("suspicious", "MFA failures: denied, no response, invalid passcode",
     {("Cisco/Duo", "suspicious")}),
    ("suspicious", "EDR detection records (behaviors, tactic and technique, IOC, severity, detection ids)",
     {("Crowdstrike/Falcon", "suspicious")}),
    ("suspicious", "web application firewall blocks (act=DENY) of malformed or attack requests",
     {("Barracuda/Barracuda WAF", "suspicious")}),
    ("malicious", "firewall deny records and network flow records of attack sessions that arrive without vendor and product metadata (the dataset's historical attack traffic)",
     {("/", "malicious")}),
)

GENERIC = {
    "benign": "routine operational activity with no security concern",
    "suspicious": "a security control flagged or blocked something, or the event otherwise deserves analyst review, but nothing shows an attack succeeded",
    "malicious": "evidence of actual hostile activity such as exploitation, malware, command-and-control or attack traffic",
}

_INTRO = "You are a SOC analyst labeling security log events, one at a time. Pick exactly one label: benign, suspicious or malicious."
_NOTES = """Notes:
- Logs are anonymized: placeholders such as <USER>, <ORG>, <CRED>, <HOST> replace names, words and numbers, so text can look garbled.
- Dates and times are masked as <DATE>, <TIME>, <EPOCH>, <YEAR>, <DAY>.
- Empty fields only mean the metadata is missing, unless a label description above says otherwise.
- When nothing was flagged, failed or blocked by a security control, the event is benign. If unsure between benign and an alert, answer benign.

Reply with one line of JSON only: {"label": "benign|suspicious|malicious", "reason": "<at most 15 words>"}"""


def system_prompt(variant: str = "codebook", excluded: set | None = None) -> str:
    """``codebook``: train-derived descriptions (minus lines read only off ``excluded`` cells); ``generic``: plain definitions."""

    lines = [_INTRO, ""]
    for label in LABELS:
        lines.append(f"{label}: {GENERIC[label]}.")
        if variant == "codebook":
            for line_label, text, cells in CODEBOOK:
                if line_label == label and (excluded is None or not cells <= excluded):
                    lines.append(f"  - {text}")
    lines += ["", _NOTES]
    return "\n".join(lines)


def user_prompt(pipeline: str, vendor: str, product: str, message: str) -> str:
    message = message if len(message) <= MAX_MESSAGE_CHARS else message[:MAX_MESSAGE_CHARS] + " ...[truncated]"
    return (f"pipeline: {pipeline or '(empty)'}\nvendor: {vendor or '(empty)'}\nproduct: {product or '(empty)'}\n"
            f"message:\n{message or '(empty)'}")


_JSON = re.compile(r"\{[^{}]*\"label\"[^{}]*\}", re.DOTALL)
_WORD = re.compile(r"\b(benign|suspicious|malicious)\b", re.IGNORECASE)


def parse_reply(text: str) -> str | None:
    for candidate in reversed(_JSON.findall(text)):
        try:
            label = str(json.loads(candidate).get("label", "")).strip().lower()
        except json.JSONDecodeError:
            continue
        if label in LABELS:
            return label
    words = _WORD.findall(text)
    return words[-1].lower() if words else None


def request_key(system: str, user: str) -> str:
    return hashlib.sha1(f"{PROMPT_VERSION}\0{system}\0{user}".encode("utf-8")).hexdigest()


_LABEL_OPEN = re.compile(r'"label"\s*:\s*"$')


def label_probabilities(completion) -> dict | None:
    """P(benign / suspicious / malicious) at the token that opens the JSON label value.

    The top-5 candidates there are matched to the labels by prefix ("ben" → benign)
    and renormalized; None when the reply has no such position.
    """

    if not completion.logprobs:
        return None
    text = ""
    for token_id, candidates in zip(completion.token_ids, completion.logprobs):
        if _LABEL_OPEN.search(text):
            mass = dict.fromkeys(LABELS, 0.0)
            for candidate in candidates.values():
                piece = (candidate.decoded_token or "").strip().lower()
                for label in LABELS:
                    if piece and label.startswith(piece):
                        mass[label] += float(np.exp(candidate.logprob))
                        break
            total = sum(mass.values())
            return {label: value / total for label, value in mass.items()} if total > 0 else None
        text += candidates[token_id].decoded_token or ""
    return None


class LLMRunner:
    """Generate replies with vLLM, caching each (system, user) pair in a JSONL file."""

    def __init__(self, cache_name: str, model: str = "qwen3.5-4b", max_tokens: int = 96):
        LLM_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "" if model == "qwen3.5-4b" else f"__{model}"
        self.cache_path = LLM_DIR / f"{cache_name}{suffix}.jsonl"
        self.model, self.max_tokens = model, max_tokens
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                self.cache[record["key"]] = record
        self._llm = None

    def _engine(self):
        if self._llm is None:
            import os

            # vLLM's distributed init picks a port in the ephemeral range, where this host's
            # proxy connections can hold it; pin one above that range.
            os.environ.setdefault("VLLM_PORT", "62411")
            from vllm import LLM

            if self.model == "qwen3.5-4b":
                # Sized for a 15 GB T4 next to 8 GB of fp16 weights: Qwen3.5 keeps a recurrent
                # state per sequence in its linear-attention layers and its prefill kernel needs
                # scratch memory vLLM's profiling misses, hence few sequences and small chunks.
                # vLLM turns prefix caching off for this hybrid architecture.
                extra = dict(max_num_seqs=32, max_num_batched_tokens=1024, limit_mm_per_prompt={"image": 0, "video": 0})
            else:
                # Plain attention: prefix caching reuses the shared system prompt across requests.
                extra = dict(max_num_seqs=64, max_num_batched_tokens=4096, enable_prefix_caching=True)
            self._llm = LLM(model=MODELS[self.model], dtype="float16", max_model_len=4096,
                            gpu_memory_utilization=0.80, seed=0, **extra)
        return self._llm

    def run(self, requests: list[tuple[str, str]], chunk: int = 256, log=print) -> list[dict]:
        """Replies for (system, user) pairs, in order; only uncached pairs are generated."""

        keys = [request_key(system, user) for system, user in requests]
        todo = {}
        for key, request in zip(keys, requests):
            if key not in self.cache:
                todo.setdefault(key, request)
        if todo:
            from vllm import SamplingParams

            params = SamplingParams(temperature=0.0, max_tokens=self.max_tokens, seed=0, logprobs=5)
            items = list(todo.items())
            log(f"LLM: {len(keys) - len(todo)} cached, {len(items)} to generate")
            with self.cache_path.open("a", encoding="utf-8") as handle:
                for start in range(0, len(items), chunk):
                    batch = items[start : start + chunk]
                    conversations = [[{"role": "system", "content": s}, {"role": "user", "content": u}] for _, (s, u) in batch]
                    outputs = self._engine().chat(conversations, params, use_tqdm=False,
                                                  chat_template_kwargs={"enable_thinking": False})
                    for (key, _), output in zip(batch, outputs):
                        completion = output.outputs[0]
                        text = completion.text
                        record = {"key": key, "text": text, "label": parse_reply(text),
                                  "probs": label_probabilities(completion),
                                  "prompt_tokens": len(output.prompt_token_ids), "output_tokens": len(completion.token_ids)}
                        self.cache[key] = record
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    log(f"LLM: {min(start + chunk, len(items))}/{len(items)} generated")
        return [self.cache[key] for key in keys]


def requests_for(frame: pd.DataFrame, variant: str = "codebook", excluded: set | None = None) -> list[tuple[str, str]]:
    """(system, user) pairs for rows with llm_message, pipeline, vendor_name, product_name."""

    system = system_prompt(variant, excluded)
    return [(system, user_prompt(p, v, pr, m)) for p, v, pr, m in
            zip(frame["pipeline"], frame["vendor_name"], frame["product_name"], frame["llm_message"])]
