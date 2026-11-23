"""Decoding packed logits, metrics and temperature calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import CHOICE, IGNORE_INDEX, NOUL, SCORE, TYPE_NAMES


@dataclass
class Record:
    kind: int
    label: int
    logits: np.ndarray  # raw logits over the question's valid options


def unpack_logits(packed) -> list[list[Record]]:
    """Inverse of ``DecisionModel._pack``: one list of :class:`Record` per example."""
    packed = np.asarray(packed, dtype=np.float64)
    out = []
    for row in packed:
        n_q, k = int(row[0]), int(row[1])
        body = row[2 : 2 + n_q * (3 + k)].reshape(n_q, 3 + k)
        recs = []
        for q in range(n_q):
            n = int(body[q, 2])
            if n == 0:  # padding question
                continue
            recs.append(Record(int(body[q, 0]), int(body[q, 1]), body[q, 3 : 3 + n].copy()))
        out.append(recs)
    return out


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def summarize(records: list[Record], temperatures: dict | None = None, n_bins: int = 15) -> dict:
    """Accuracy, NLL and expected calibration error, overall and per question type."""
    temperatures = temperatures or {}
    stats: dict[str, dict[str, list]] = {}
    for r in records:
        if r.label == IGNORE_INDEX:
            continue
        name = TYPE_NAMES[r.kind]
        p = _softmax(r.logits, temperatures.get(name, 1.0))
        conf = float(p.max())
        s = stats.setdefault(name, {"correct": [], "nll": [], "conf": []})
        s["correct"].append(float(p.argmax() == r.label))
        s["nll"].append(float(-np.log(max(p[r.label], 1e-12))))
        s["conf"].append(conf)

    def ece(correct, conf):
        correct, conf = np.asarray(correct), np.asarray(conf)
        bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
        total = 0.0
        for b in range(n_bins):
            m = bins == b
            if m.any():
                total += m.mean() * abs(correct[m].mean() - conf[m].mean())
        return float(total)

    metrics: dict[str, float] = {}
    for name, s in sorted(stats.items()):
        metrics[f"accuracy_{name}"] = float(np.mean(s["correct"]))
    if stats:
        c = [x for s in stats.values() for x in s["correct"]]
        n = [x for s in stats.values() for x in s["nll"]]
        f = [x for s in stats.values() for x in s["conf"]]
        metrics["accuracy"] = float(np.mean(c))
        metrics["nll"] = float(np.mean(n))
        metrics["ece"] = ece(c, f)
    return metrics


def compute_metrics(eval_pred) -> dict:
    """Drop-in ``compute_metrics`` for ``Trainer``: decodes packed logits and reports accuracy / NLL / ECE."""
    predictions = eval_pred.predictions if hasattr(eval_pred, "predictions") else eval_pred[0]
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    records = [r for example in unpack_logits(predictions) for r in example]
    return summarize(records)


def fit_temperatures(records: list[Record], min_samples: int = 5) -> dict[str, float]:
    """Fit one temperature per question type by minimising NLL (post-hoc calibration)."""
    temps: dict[str, float] = {}
    for kind in (CHOICE, NOUL, SCORE):
        recs = [r for r in records if r.kind == kind and r.label != IGNORE_INDEX]
        if len(recs) < min_samples:
            continue
        k = max(len(r.logits) for r in recs)
        logits = torch.full((len(recs), k), -1e9, dtype=torch.float64)
        for i, r in enumerate(recs):
            logits[i, : len(r.logits)] = torch.as_tensor(r.logits)
        labels = torch.tensor([r.label for r in recs])
        log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(logits / log_t.clamp(-3, 3).exp(), labels)
            loss.backward()
            return loss

        opt.step(closure)
        temps[TYPE_NAMES[kind]] = float(log_t.detach().clamp(-3, 3).exp())
    return temps
