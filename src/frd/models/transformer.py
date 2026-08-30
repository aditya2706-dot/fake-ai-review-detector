"""Optional transformer text stage (DistilBERT / DeBERTa fine-tune).

WHEN THIS IS WORTH IT
---------------------
TF-IDF plus a linear model is a strong, fast, interpretable text baseline, and
on template-driven review spam it is hard to beat -- reused phrasing is exactly
what n-grams capture. A transformer earns its keep in two specific situations:

  * LLM-generated reviews. These are fluent, varied, and template-free, so
    n-gram overlap is low. What gives them away is distributional: unnaturally
    even syntax, low burstiness, an absence of the concrete oddities real
    experience produces. A pretrained encoder picks that up; TF-IDF cannot.
  * Multilingual or code-switched marketplaces, where per-language TF-IDF
    vocabularies fragment and a multilingual encoder shares representation.

WHEN IT IS NOT
--------------
If PR-AUC is already dominated by behavioural features -- which the ablation in
this project shows it is -- a transformer adds GPU cost and latency for a small
gain. Measure the fused ablation first; add this only if text is your binding
constraint.

INSTALL
-------
    pip install -e ".[deep]"

Everything is imported lazily so the core pipeline never depends on torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import LABEL, TEXT, TITLE

DEFAULT_MODEL = "distilbert-base-uncased"


@dataclass
class TransformerConfig:
    model_name: str = DEFAULT_MODEL
    max_length: int = 256
    batch_size: int = 32
    epochs: int = 2
    learning_rate: float = 3e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42
    output_dir: str = "artifacts/models/transformer"
    fp16: bool = False


def _require_deps():
    try:
        import torch  # noqa: F401, PLC0415
        import transformers  # noqa: F401, PLC0415
    except ImportError as e:  # pragma: no cover - env dependent
        raise ImportError(
            "the transformer stage needs extra packages:\n"
            '    pip install -e ".[deep]"'
        ) from e


def _compose(df: pd.DataFrame) -> list[str]:
    """Title and body as one sequence; titles are short and carry signal."""
    title = df[TITLE].fillna("").astype(str) if TITLE in df.columns else ""
    body = df[TEXT].fillna("").astype(str)
    return (title + " [SEP] " + body).tolist() if len(title) else body.tolist()


class TransformerTextScorer:
    """Fine-tuned sequence classifier producing P(fake) from text alone.

    Slots into the stacked architecture in place of -- or alongside -- the
    TF-IDF stage A: its output is one more scalar column for the stage-B
    booster, so the fusion model needs no changes.
    """

    def __init__(self, cfg: TransformerConfig | None = None):
        self.cfg = cfg or TransformerConfig()
        self.model = None
        self.tokenizer = None

    def fit(
        self, train: pd.DataFrame, val: pd.DataFrame | None = None
    ) -> "TransformerTextScorer":
        _require_deps()
        import torch  # noqa: PLC0415
        from torch.utils.data import DataLoader, TensorDataset  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification, AutoTokenizer,
            get_linear_schedule_with_warmup,
        )

        torch.manual_seed(self.cfg.seed)
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name, num_labels=2
        ).to(device)

        enc = self.tokenizer(
            _compose(train), truncation=True, padding="max_length",
            max_length=self.cfg.max_length, return_tensors="pt",
        )
        y = torch.tensor(train[LABEL].to_numpy(dtype=np.int64))
        loader = DataLoader(
            TensorDataset(enc["input_ids"], enc["attention_mask"], y),
            batch_size=self.cfg.batch_size, shuffle=True,
        )

        # Class imbalance handled in the loss, mirroring the sample weighting
        # used by the gradient-boosted stage.
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        weights = torch.tensor(
            [1.0, neg / max(pos, 1.0)], dtype=torch.float32, device=device
        )
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        total = len(loader) * self.cfg.epochs
        sched = get_linear_schedule_with_warmup(
            opt, int(total * self.cfg.warmup_ratio), total
        )

        self.model.train()
        for epoch in range(self.cfg.epochs):
            running = 0.0
            for step, (ids, mask, labels) in enumerate(loader):
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                opt.zero_grad()
                logits = self.model(input_ids=ids, attention_mask=mask).logits
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                running += float(loss.item())
                if step % 50 == 0:
                    print(f"  epoch {epoch + 1} step {step}/{len(loader)} "
                          f"loss {running / (step + 1):.4f}")
            if val is not None:
                from sklearn.metrics import average_precision_score  # noqa: PLC0415
                ap = average_precision_score(
                    val[LABEL].to_numpy(int), self.predict_proba(val)
                )
                print(f"  epoch {epoch + 1}: val PR-AUC {ap:.4f}")
                self.model.train()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        _require_deps()
        import torch  # noqa: PLC0415

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("model is not fitted")
        device = next(self.model.parameters()).device
        self.model.eval()

        texts = _compose(X)
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.cfg.batch_size):
                batch = texts[i: i + self.cfg.batch_size]
                enc = self.tokenizer(
                    batch, truncation=True, padding=True,
                    max_length=self.cfg.max_length, return_tensors="pt",
                ).to(device)
                logits = self.model(**enc).logits
                out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def save(self, path: str | Path | None = None) -> Path:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        p = Path(path or self.cfg.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(p)
        self.tokenizer.save_pretrained(p)
        return p

    @classmethod
    def load(cls, path: str | Path,
             cfg: TransformerConfig | None = None) -> "TransformerTextScorer":
        _require_deps()
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification, AutoTokenizer,
        )
        obj = cls(cfg)
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path)
        return obj
