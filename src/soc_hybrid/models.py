"""Classifier candidates with one interface: fit(docs, labels, weights) / predict_proba(docs).

Probabilities come back in LABELS order (benign, suspicious, malicious).
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import LABELS

MAX_DOC_CHARS = 6000


def label_codes(labels) -> np.ndarray:
    index = {label: i for i, label in enumerate(LABELS)}
    return np.array([index[label] for label in labels], dtype=int)


def training_weights(unique: pd.DataFrame, scheme: str = "doc") -> np.ndarray:
    """Per unique document weight, then rescaled so every class carries the same total.

    ``doc``: each distinct document counts once (duplicates carry no extra
    weight); ``log``: 1 + log(rows); ``rows``: every event counts.
    """

    rows = unique["rows"].to_numpy(dtype=float)
    base = {"doc": np.ones_like(rows), "log": 1.0 + np.log(rows), "rows": rows}[scheme]
    labels = unique["label"].to_numpy()
    weights = base.copy()
    present = [label for label in LABELS if (labels == label).any()]
    for label in present:
        mask = labels == label
        weights[mask] *= base.sum() / (len(present) * base[mask].sum())
    return weights / weights.mean()


def _clip(docs) -> list[str]:
    return [doc[:MAX_DOC_CHARS] for doc in docs]


class TfidfLogReg:
    """TF-IDF n-grams + multinomial logistic regression."""

    def __init__(self, analyzer: str = "word", ngram_range=(1, 2), max_features: int = 200_000,
                 min_df: int = 2, C: float = 10.0, max_iter: int = 300):
        token_pattern = r"(?u)\b[A-Za-z_][A-Za-z0-9_]+\b|0"
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer, ngram_range=ngram_range, max_features=max_features, min_df=min_df,
            sublinear_tf=True, lowercase=True, dtype=np.float32,
            **({"token_pattern": token_pattern} if analyzer == "word" else {}),
        )
        self.C = C
        self.max_iter = max_iter
        self.model = None

    def fit(self, docs, labels, weights=None):
        x = self.vectorizer.fit_transform(_clip(docs))
        self.model = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs")
        self.model.fit(x, label_codes(labels), sample_weight=weights)
        return self

    def predict_proba(self, docs) -> np.ndarray:
        x = self.vectorizer.transform(_clip(docs))
        proba = np.zeros((x.shape[0], len(LABELS)))
        proba[:, self.model.classes_] = self.model.predict_proba(x)
        return proba


_CNN_TOKEN = re.compile(r"[a-z_]+|0|[^\sa-z0-9_]")


def cnn_tokens(doc: str) -> list[str]:
    return _CNN_TOKEN.findall(doc.lower())


class TextCNN:
    """Word-level CNN (Kim 2014): embeddings learned from scratch, filters of width 2-5, max-pool.

    A token sequence keeps the head (source fields + message start) and the tail
    of long documents, where JSON logs put their outcome fields.
    """

    def __init__(self, embed_dim: int = 96, filters: int = 128, widths=(2, 3, 4, 5), head: int = 320,
                 tail: int = 64, max_vocab: int = 60_000, epochs: int = 5, batch_size: int = 256,
                 lr: float = 2e-3, dropout: float = 0.3, seed: int = 0, device: str = "cuda", threads: int | None = None):
        self.embed_dim, self.filters, self.widths = embed_dim, filters, tuple(widths)
        self.head, self.tail, self.max_vocab = head, tail, max_vocab
        self.epochs, self.batch_size, self.lr, self.dropout = epochs, batch_size, lr, dropout
        self.seed, self.device, self.threads = seed, device, threads
        self.vocab: dict[str, int] = {}
        self.net = None

    def _encode(self, docs) -> np.ndarray:
        length = self.head + self.tail
        out = np.zeros((len(docs), length), dtype=np.int64)
        for i, doc in enumerate(docs):
            tokens = cnn_tokens(doc[:MAX_DOC_CHARS * 2])
            if len(tokens) > length:
                tokens = tokens[: self.head] + tokens[-self.tail:]
            ids = [self.vocab.get(token, 1) for token in tokens]
            out[i, : len(ids)] = ids
        return out

    def _build(self, torch):
        nn = torch.nn
        widths, filters, embed_dim, dropout = self.widths, self.filters, self.embed_dim, self.dropout
        vocab_size = len(self.vocab) + 2

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.convs = nn.ModuleList(nn.Conv1d(embed_dim, filters, w, padding=w - 1) for w in widths)
                self.drop = nn.Dropout(dropout)
                self.out = nn.Linear(filters * len(widths), len(LABELS))

            def forward(self, ids):
                mask = (ids > 0).float().unsqueeze(1)
                x = self.embed(ids).transpose(1, 2)
                pooled = []
                for conv in self.convs:
                    h = torch.relu(conv(x))[:, :, : ids.shape[1]]
                    pooled.append((h * mask - (1 - mask) * 1e4).max(dim=2).values.float())
                return self.out(self.drop(torch.cat(pooled, dim=1)))

        return Net()

    def fit(self, docs, labels, weights=None):
        import torch

        if self.threads:
            torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        counts = Counter()
        for doc in docs:
            counts.update(set(cnn_tokens(doc[:MAX_DOC_CHARS * 2])))
        common = [token for token, n in counts.most_common(self.max_vocab) if n >= 2]
        self.vocab = {token: i + 2 for i, token in enumerate(common)}
        ids = torch.from_numpy(self._encode(docs))
        y = torch.from_numpy(label_codes(labels))
        w = torch.from_numpy(np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)).float()
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self.net = self._build(torch).to(device)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        steps = self.epochs * int(np.ceil(len(y) / self.batch_size))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=steps)
        amp = device.type == "cuda"  # fp16 autocast on the T4's tensor cores
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        self.net.train()
        for _ in range(self.epochs):
            order = torch.from_numpy(rng.permutation(len(y)))
            for start in range(0, len(y), self.batch_size):
                batch = order[start : start + self.batch_size]
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = self.net(ids[batch].to(device))
                    loss = torch.nn.functional.cross_entropy(logits.float(), y[batch].to(device), reduction="none")
                    loss = (loss * w[batch].to(device)).sum() / w[batch].sum().to(device)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
        self.net.eval()
        self.device_ = device
        return self

    def predict_proba(self, docs) -> np.ndarray:
        import torch

        out = []
        amp = self.device_.type == "cuda"
        with torch.no_grad(), torch.autocast(device_type=self.device_.type, dtype=torch.float16, enabled=amp):
            for start in range(0, len(docs), 2048):
                ids = torch.from_numpy(self._encode(docs[start : start + 2048])).to(self.device_)
                out.append(torch.softmax(self.net(ids).float(), dim=1).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, len(LABELS)))

    def __getstate__(self):
        # The network class is built lazily (torch is optional), so pickle its weights, not the module.
        state = self.__dict__.copy()
        net = state.pop("net", None)
        state["net_state"] = None if net is None else {k: v.cpu() for k, v in net.state_dict().items()}
        state["device_"] = None
        return state

    def __setstate__(self, state):
        net_state = state.pop("net_state", None)
        self.__dict__.update(state)
        self.net = None
        if net_state is not None:
            import torch

            self.device_ = torch.device(self.device if torch.cuda.is_available() else "cpu")
            self.net = self._build(torch)
            self.net.load_state_dict(net_state)
            self.net.to(self.device_).eval()


def make_classifier(name: str, **kwargs):
    if name == "tfidf_word_lr":
        return TfidfLogReg(analyzer="word", ngram_range=(1, 2), **kwargs)
    if name == "tfidf_char_lr":
        return TfidfLogReg(analyzer="char_wb", ngram_range=(3, 5), max_features=300_000, **kwargs)
    if name == "textcnn":
        return TextCNN(**kwargs)
    raise ValueError(f"unknown classifier {name!r}")
