"""``pipeline("choice" | "noul" | "score" | "decision", model=...)`` for inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from .data import CHOICE, NOUL, SCORE, DataCollatorForDecision, DecisionEncoder, Question, parse_questions
from .metrics import _softmax
from .modeling import AutoModelForDecision, DecisionModel

TASKS = ("choice", "noul", "score", "decision")
_DEFAULT_INSTRUCTIONS = {
    "choice": "Which option fits best?",
    "noul": "Is the statement true?",
    "score": "How high is the level?",
}


class DecisionPipeline:
    """Runs a decision model on raw text.

    * ``"choice"``:  ``pipe(text, options=["a", "b"])`` (or ``{key: description}``)
    * ``"noul"``:    ``pipe(text, instructions="Does the customer want a refund?")``
    * ``"score"``:   ``pipe(text, levels=["low", "medium", "high"])``
    * ``"decision"``: ``pipe(text, questions={name: Choice(...) | Noul(...) | Score(...)})`` -> one pass, many answers

    A single string returns one result, a list of strings returns a list.  Probabilities are
    temperature-scaled with ``model.config.temperatures`` (see :func:`decisionhead.calibrate`).
    """

    def __init__(self, task, model: DecisionModel, tokenizer, max_length=None, batch_size=8, device=None, **defaults):
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}; choose one of {TASKS}.")
        self.task, self.model, self.tokenizer = task, model, tokenizer
        self.batch_size = batch_size
        self.defaults = defaults
        if device is not None:
            self.model.to(device)
        self.encoder = DecisionEncoder(tokenizer, max_length=max_length or model.config.max_input_length)
        self.collator = DataCollatorForDecision(tokenizer)
        self.model.eval()

    # ------------------------------------------------------------------ public
    def __call__(self, inputs, **kwargs):
        opts = {**self.defaults, **kwargs}
        single = isinstance(inputs, str)
        states = [inputs] if single else list(inputs)
        question_sets = [self._questions_for(opts) for _ in states]
        results = self._run(states, question_sets)
        formatted = [self._format(res, qs) for res, qs in zip(results, question_sets)]
        return formatted[0] if single else formatted

    # ------------------------------------------------------------------ internals
    def _questions_for(self, opts: dict) -> list[Question]:
        if self.task == "decision":
            questions = opts.get("questions")
            if questions is None:
                raise ValueError("pipeline('decision') needs questions={...}.")
            return parse_questions(questions)
        name = opts.get("name", self.task)
        instructions = opts.get("instructions") or _DEFAULT_INSTRUCTIONS[self.task]
        if self.task == "noul":
            spec = {"type": "noul", "instructions": instructions}
        elif self.task == "choice":
            options = opts.get("options") or opts.get("criteria")
            if options is None:
                raise ValueError("pipeline('choice') needs options=[...] (or a {key: description} dict).")
            spec = {"type": "choice", "instructions": instructions, "criteria": options}
        else:
            levels = opts.get("levels") or opts.get("criteria")
            if levels is None:
                raise ValueError("pipeline('score') needs levels=[...] ordered from low to high.")
            spec = {"type": "score", "instructions": instructions, "criteria": levels}
        return parse_questions({name: spec})

    @torch.no_grad()
    def _run(self, states: list[str], question_sets: list[list[Question]]) -> list[list[np.ndarray]]:
        """Return, per example, one probability vector per question."""
        temps = self.model.config.temperatures
        built = [self.encoder.build(s, qs) for s, qs in zip(states, question_sets)]
        out: list[list[np.ndarray]] = []
        for start in range(0, len(built), self.batch_size):
            chunk = built[start : start + self.batch_size]
            batch = self.collator([f for f, _ in chunk])
            batch.pop("labels")
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            logits = self.model.decision_logits(**batch).float().cpu().numpy()
            for b, (_, qs) in enumerate(chunk):
                probs = []
                for qi, q in enumerate(qs):
                    n = q.num_logits
                    probs.append(_softmax(logits[b, qi, :n].astype(np.float64), temps.get(q.type, 1.0)))
                out.append(probs)
        return out

    def _format(self, probs: list[np.ndarray], qs: list[Question]):
        results = {q.name: self._format_one(p, q) for p, q in zip(probs, qs)}
        return results if self.task == "decision" else next(iter(results.values()))

    @staticmethod
    def _format_one(p: np.ndarray, q: Question) -> dict:
        top = int(p.argmax())
        if q.kind == NOUL:
            return {"label": bool(top == 1), "score": float(p[top]), "probability": float(p[1])}
        probs = {k: float(v) for k, v in zip(q.keys, p)}
        res = {"label": q.keys[top], "score": float(p[top]), "probabilities": probs}
        if q.kind == SCORE:
            res["index"] = top
            res["expected"] = float((p * np.arange(len(p))).sum())
        return res


def pipeline(task: str, model: Any = None, tokenizer=None, **kwargs) -> DecisionPipeline:
    """Create a :class:`DecisionPipeline`.

    ``model`` is a path/Hub id of a saved decision model, or an in-memory ``DecisionModel``
    (then pass ``tokenizer=`` too).  Remaining kwargs are defaults for every call, e.g.
    ``pipeline("choice", model=..., options=["billing", "technical"])``.
    """
    if model is None:
        raise ValueError("pipeline() needs model=<path or DecisionModel>.")
    if isinstance(model, str):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForDecision.from_pretrained(model)
    elif tokenizer is None:
        raise ValueError("Pass tokenizer=... together with an in-memory model.")
    if not isinstance(model, DecisionModel):
        raise TypeError("model must be a DecisionModel or a path to one saved with save_pretrained().")
    pipe_kwargs = {k: kwargs.pop(k) for k in ("max_length", "batch_size", "device") if k in kwargs}
    return DecisionPipeline(task, model, tokenizer, **pipe_kwargs, **kwargs)
