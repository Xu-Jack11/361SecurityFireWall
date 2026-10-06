"""PyTorch GPU model for sparse TF-IDF security log classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .constants import LABELS


def torch_cuda_available() -> bool:
    """Return whether PyTorch can use CUDA without making torch a hard import."""

    try:
        torch = _import_torch()
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_torch_device(requested: str = "auto", require_cuda: bool = False) -> Any:
    """Resolve a torch device and optionally require CUDA."""

    torch = _import_torch()
    normalized = (requested or "auto").lower()
    if normalized == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(normalized)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("GPU backend requires a CUDA device; use --device auto or --device cuda")
    return device


@dataclass(frozen=True)
class TorchTrainingConfig:
    max_features: int
    min_df: int
    random_state: int
    device: str = "auto"
    require_cuda: bool = False
    epochs: int = 3
    batch_size: int = 8192
    learning_rate: float = 0.05
    weight_decay: float = 1e-5


class TorchTfidfClassifier:
    """A sparse TF-IDF linear classifier trained with PyTorch on CPU or CUDA."""

    def __init__(
        self,
        max_features: int,
        min_df: int,
        random_state: int,
        device: str = "auto",
        require_cuda: bool = False,
        epochs: int = 3,
        batch_size: int = 8192,
        learning_rate: float = 0.05,
        weight_decay: float = 1e-5,
    ) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.random_state = random_state
        self.device = device
        self.require_cuda = require_cuda
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.vectorizer = _make_vectorizer(max_features, min_df)
        self.classes_: np.ndarray | None = None
        self.weight_: Any | None = None
        self.bias_: Any | None = None
        self.runtime_device_: str | None = None

    @classmethod
    def from_config(cls, config: TorchTrainingConfig) -> "TorchTfidfClassifier":
        return cls(
            max_features=config.max_features,
            min_df=config.min_df,
            random_state=config.random_state,
            device=config.device,
            require_cuda=config.require_cuda,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def fit(self, documents: pd.Series, labels: pd.Series) -> "TorchTfidfClassifier":
        torch = _import_torch()
        torch.nn.functional
        device = resolve_torch_device(self.device, self.require_cuda)
        self.runtime_device_ = str(device)
        _seed_torch(torch, self.random_state)

        y_labels = labels.astype(str).reset_index(drop=True)
        self.classes_ = np.array([label for label in LABELS if label in set(y_labels)])
        if len(self.classes_) < 2:
            raise ValueError("TorchTfidfClassifier requires at least two classes")
        label_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y = np.array([label_to_idx[label] for label in y_labels], dtype=np.int64)

        x_matrix = self.vectorizer.fit_transform(documents.astype(str).reset_index(drop=True))
        self._initialize_parameters(torch, x_matrix.shape[1], len(self.classes_), device)
        class_weights = self._class_weights(torch, y, device)
        optimizer = torch.optim.AdamW(
            [self.weight_, self.bias_],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        y_tensor = torch.as_tensor(y, dtype=torch.long, device=device)

        for _epoch in range(self.epochs):
            order = np.random.default_rng(self.random_state + _epoch).permutation(x_matrix.shape[0])
            for start in range(0, len(order), self.batch_size):
                batch_indices = order[start : start + self.batch_size]
                batch_x = _csr_to_torch_sparse(x_matrix[batch_indices], device)
                batch_y = y_tensor[torch.as_tensor(batch_indices, dtype=torch.long, device=device)]
                logits = torch.sparse.mm(batch_x, self.weight_) + self.bias_
                loss = torch.nn.functional.cross_entropy(logits, batch_y, weight=class_weights)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, documents: pd.Series) -> np.ndarray:
        torch = _import_torch()
        self._require_fitted()
        device = resolve_torch_device(self.runtime_device_ or self.device, require_cuda=False)
        self._move_parameters(torch, device)

        x_matrix = self.vectorizer.transform(documents.astype(str).reset_index(drop=True))
        probabilities: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, x_matrix.shape[0], self.batch_size):
                batch_x = _csr_to_torch_sparse(x_matrix[start : start + self.batch_size], device)
                logits = torch.sparse.mm(batch_x, self.weight_) + self.bias_
                batch_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                probabilities.append(batch_probs)
        if not probabilities:
            return np.empty((0, len(self.classes_)), dtype=np.float32)
        return np.vstack(probabilities)

    def predict(self, documents: pd.Series) -> np.ndarray:
        probabilities = self.predict_proba(documents)
        return self.classes_[probabilities.argmax(axis=1)]

    @property
    def coef_(self) -> np.ndarray:
        self._require_fitted()
        return self.weight_.detach().cpu().numpy().T

    def get_feature_names_out(self) -> np.ndarray:
        return self.vectorizer.get_feature_names_out()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if state.get("weight_") is not None:
            state["weight_"] = state["weight_"].detach().cpu()
        if state.get("bias_") is not None:
            state["bias_"] = state["bias_"].detach().cpu()
        state["runtime_device_"] = "cpu"
        return state

    def _initialize_parameters(self, torch: Any, input_dim: int, num_classes: int, device: Any) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_state)
        initial_weight = torch.empty((input_dim, num_classes), dtype=torch.float32)
        torch.nn.init.xavier_uniform_(initial_weight, generator=generator)
        self.weight_ = initial_weight.to(device).requires_grad_(True)
        self.bias_ = torch.zeros(num_classes, dtype=torch.float32, device=device, requires_grad=True)

    def _class_weights(self, torch: Any, y: np.ndarray, device: Any) -> Any:
        counts = np.bincount(y, minlength=len(self.classes_)).astype(np.float32)
        weights = len(y) / (len(self.classes_) * np.maximum(counts, 1.0))
        return torch.as_tensor(weights, dtype=torch.float32, device=device)

    def _move_parameters(self, torch: Any, device: Any) -> None:
        if str(device) == self.runtime_device_:
            return
        self.weight_ = self.weight_.detach().to(device).requires_grad_(False)
        self.bias_ = self.bias_.detach().to(device).requires_grad_(False)
        self.runtime_device_ = str(device)

    def _require_fitted(self) -> None:
        if self.classes_ is None or self.weight_ is None or self.bias_ is None:
            raise RuntimeError("TorchTfidfClassifier is not fitted")


def _make_vectorizer(max_features: int, min_df: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        max_features=max_features,
        min_df=min_df,
        max_df=0.98,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b[\w.\-/:=@+]+\b",
    )


def _csr_to_torch_sparse(matrix: sparse.spmatrix, device: Any) -> Any:
    torch = _import_torch()
    csr = matrix.tocsr()
    coo = csr.tocoo()
    if coo.nnz == 0:
        indices = torch.empty((2, 0), dtype=torch.long, device=device)
        values = torch.empty((0,), dtype=torch.float32, device=device)
    else:
        indices = torch.as_tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device)
        values = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, size=coo.shape, device=device).coalesce()


def _seed_torch(torch: Any, random_state: int) -> None:
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for GPU acceleration. Install it with "
            "`uv pip install torch` or run `uv sync --extra gpu`."
        ) from exc
    return torch

