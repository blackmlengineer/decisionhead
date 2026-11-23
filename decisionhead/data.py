"""Question specs, example encoding and batching for decision models.

Data format
-----------
One example is ``{"state": str, "questions": {...}, "gold": {...}}``.

``questions`` maps a question name to a spec:

* ``{"type": "choice", "instructions": ..., "criteria": {key: description}}``  (or a list of keys)
* ``{"type": "noul",   "instructions": ...}``                                  (yes / no)
* ``{"type": "score",  "instructions": ..., "criteria": [level0, level1, ...]}`` (ordered, low -> high)

``gold`` maps a question name to the answer: the option key (choice), ``True``/``False``
(noul) or the level index / label (score).  Missing gold entries are ignored by the loss.

``questions`` and ``gold`` may be dicts or JSON strings.  Prefer JSON strings inside a
``datasets.Dataset`` (see :func:`make_example`): Arrow cannot hold dicts whose keys differ
between rows without padding them with nulls.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

CHOICE, NOUL, SCORE = 0, 1, 2
TYPE_IDS = {"choice": CHOICE, "noul": NOUL, "score": SCORE}
TYPE_NAMES = {v: k for k, v in TYPE_IDS.items()}
_ALIASES = {
    "binary": "noul",
    "boolean": "noul",
    "bool": "noul",
    "yes_no": "noul",
    "ordinal": "score",
    "scale": "score",
    "categorical": "choice",
    "select": "choice",
}
IGNORE_INDEX = -100


# --------------------------------------------------------------------------- spec helpers
@dataclass
class Choice:
    """Pick exactly one of ``criteria`` (a ``{key: description}`` dict or a list of keys)."""

    instructions: str
    criteria: dict | list
    type: ClassVar[str] = "choice"

    def to_dict(self) -> dict:
        return {"type": self.type, "instructions": self.instructions, "criteria": self.criteria}


@dataclass
class Noul:
    """A yes / no question."""

    instructions: str
    type: ClassVar[str] = "noul"

    def to_dict(self) -> dict:
        return {"type": self.type, "instructions": self.instructions}


@dataclass
class Score:
    """Pick a level from an ordered list ``criteria`` (lowest first)."""

    instructions: str
    criteria: list
    type: ClassVar[str] = "score"

    def to_dict(self) -> dict:
        return {"type": self.type, "instructions": self.instructions, "criteria": list(self.criteria)}


@dataclass
class Question:
    name: str
    kind: int  # CHOICE / NOUL / SCORE
    instructions: str
    keys: list[str]  # option keys / level labels (empty for noul)
    descriptions: list[str]  # option descriptions ("" when absent)

    @property
    def type(self) -> str:
        return TYPE_NAMES[self.kind]

    @property
    def num_logits(self) -> int:
        return 2 if self.kind == NOUL else len(self.keys)


def _maybe_json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, (str, bytes)) else value


def parse_questions(questions: Any) -> list[Question]:
    """Normalize any accepted ``questions`` representation into a list of :class:`Question`."""
    if isinstance(questions, (list, tuple)) and questions and all(isinstance(q, Question) for q in questions):
        return list(questions)  # already parsed
    questions = _maybe_json(questions)
    if isinstance(questions, dict):
        items = list(questions.items())
    elif isinstance(questions, (list, tuple)):
        items = []
        for q in questions:
            q = q.to_dict() if hasattr(q, "to_dict") else q
            if "name" not in q:
                raise ValueError("When 'questions' is a list every entry needs a 'name'.")
            items.append((q["name"], q))
    else:
        raise TypeError(f"'questions' must be a dict, list or JSON string, got {type(questions).__name__}.")
    if not items:
        raise ValueError("At least one question is required.")

    parsed = []
    for name, spec in items:
        spec = dict(spec.to_dict() if hasattr(spec, "to_dict") else spec)
        qtype = str(spec.get("type", "")).lower()
        qtype = _ALIASES.get(qtype, qtype)
        if qtype not in TYPE_IDS:
            raise ValueError(f"Question {name!r}: unknown type {spec.get('type')!r}; use choice, noul or score.")
        instructions = str(spec.get("instructions") or spec.get("question") or "")
        if qtype == "noul":
            keys, descs = [], []
        else:
            crit = spec.get("criteria", spec.get("options", spec.get("levels")))
            crit = _maybe_json(crit) if crit is not None else None
            if crit is None:
                raise ValueError(f"Question {name!r} of type {qtype!r} needs 'criteria'.")
            if isinstance(crit, dict):
                keys = [str(k) for k in crit]
                descs = ["" if v is None else str(v) for v in crit.values()]
            else:
                keys = [str(k) for k in crit]
                descs = [""] * len(keys)
            if len(keys) < 2:
                raise ValueError(f"Question {name!r} needs at least two options, got {keys}.")
            if len(set(keys)) != len(keys):
                raise ValueError(f"Question {name!r} has duplicate options: {keys}.")
        parsed.append(Question(str(name), TYPE_IDS[qtype], instructions, keys, descs))
    if len({q.name for q in parsed}) != len(parsed):
        raise ValueError("Question names must be unique.")
    return parsed


_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}


def parse_label(q: Question, gold: Any) -> int:
    """Turn a gold answer into the index the loss expects (``IGNORE_INDEX`` if missing)."""
    if gold is None:
        return IGNORE_INDEX
    if q.kind == NOUL:
        if isinstance(gold, bool):
            return int(gold)
        s = str(gold).strip().lower()
        if s in _TRUE:
            return 1
        if s in _FALSE:
            return 0
        raise ValueError(f"Question {q.name!r}: gold {gold!r} is not a yes/no answer.")
    s = str(gold)
    if s in q.keys:
        return q.keys.index(s)
    if q.kind == SCORE and not isinstance(gold, bool):
        try:
            idx = int(s)
        except ValueError:
            idx = None
        if idx is not None and 0 <= idx < len(q.keys):
            return idx
    raise ValueError(f"Question {q.name!r}: gold {gold!r} is not one of {q.keys}.")


def make_example(state: str, questions: Any, gold: Any = None) -> dict:
    """Build a ``datasets``-friendly row (``questions`` / ``gold`` stored as JSON strings)."""
    questions = _maybe_json(questions)
    if isinstance(questions, dict):
        questions = {n: (s.to_dict() if hasattr(s, "to_dict") else s) for n, s in questions.items()}
    row = {"state": state, "questions": json.dumps(questions)}
    if gold is not None:
        row["gold"] = json.dumps(gold)
    return row


# --------------------------------------------------------------------------- encoding
class DecisionEncoder:
    """Tokenizes ``state`` + ``questions`` into one sequence and records where to read answers.

    Training-time augmentation (applied once per row at ``map`` time, seeded):
    ``shuffle_options`` (order of unordered choice options), ``shuffle_questions`` and
    ``question_dropout`` (probability of dropping each question, keeping at least one).

    Use it exactly like a tokenizing function with ``datasets.map``::

        encoder = DecisionEncoder(tokenizer, max_length=512)
        tokenized = dataset.map(encoder, remove_columns=dataset.column_names)

    Layout of the sequence (every piece is tokenized separately, so positions are exact)::

        State:
        <state text>
        Question (name): instructions
        Options:
        - key: description        <- last token = option read-out position
        Answer (name):            <- last token = yes/no read-out position
    """

    def __init__(
        self,
        tokenizer,
        max_length: int | None = None,
        shuffle_options: bool = False,
        shuffle_questions: bool = False,
        question_dropout: float = 0.0,
        seed: int = 0,
    ):
        self.tokenizer = tokenizer
        if max_length is None:
            mml = getattr(tokenizer, "model_max_length", None)
            max_length = min(mml, 2048) if isinstance(mml, int) and mml < 10**6 else 2048
        self.max_length = max_length
        self.shuffle_options = shuffle_options
        self.shuffle_questions = shuffle_questions
        self.question_dropout = question_dropout
        self._rng = random.Random(seed)

        cls_id = getattr(tokenizer, "cls_token_id", None)
        bos_id = getattr(tokenizer, "bos_token_id", None)
        sep_id = getattr(tokenizer, "sep_token_id", None)
        first = cls_id if cls_id is not None else bos_id
        self._prefix = [first] if first is not None else []
        self._suffix = [sep_id] if (cls_id is not None and sep_id is not None) else []

    # -- tokenization helpers
    def _ids(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _question_segments(self, q: Question):
        """Return ``(segments, option_segment_indices, answer_segment_index)``."""
        segs = [self._ids(f"\nQuestion ({q.name}): {q.instructions}\n")]
        opt_idx: list[int] = []
        if q.kind == CHOICE:
            segs.append(self._ids("Options:\n"))
        elif q.kind == SCORE:
            segs.append(self._ids("Levels, lowest to highest:\n"))
        for key, desc in zip(q.keys, q.descriptions):
            segs.append(self._ids(f"- {key}: {desc}\n" if desc else f"- {key}\n"))
            opt_idx.append(len(segs) - 1)
        segs.append(self._ids(f"Answer ({q.name}):"))
        return segs, opt_idx, len(segs) - 1

    def build(self, state: str, questions: Any, gold: Any = None) -> tuple[dict, list[Question]]:
        """Encode one example. Returns ``(features, questions_in_encoded_order)``."""
        qs = parse_questions(questions)
        gold = _maybe_json(gold) if gold is not None else {}
        if not isinstance(gold, dict):
            raise TypeError("'gold' must be a dict (or JSON object string) keyed by question name.")
        unknown = set(gold) - {q.name for q in qs}
        if unknown:
            raise ValueError(f"'gold' has answers for unknown questions: {sorted(unknown)}")

        # Augmentation (training-time only): all questions share one context, so a model trained on
        # fixed question sets can be brittle when later asked a subset / another order (e.g. by
        # pipeline("noul")). Dropping and shuffling questions makes it robust to that.
        if self.question_dropout > 0 and len(qs) > 1:
            kept = [q for q in qs if self._rng.random() >= self.question_dropout]
            qs = kept or [self._rng.choice(qs)]
        if self.shuffle_questions:
            qs = list(qs)
            self._rng.shuffle(qs)

        # Optionally shuffle the order of unordered options (choice) as augmentation.
        if self.shuffle_options:
            shuffled = []
            for q in qs:
                if q.kind == CHOICE:
                    order = list(range(len(q.keys)))
                    self._rng.shuffle(order)
                    q = Question(q.name, q.kind, q.instructions, [q.keys[i] for i in order], [q.descriptions[i] for i in order])
                shuffled.append(q)
            qs = shuffled

        head = self._ids("State:\n")
        state_ids = self._ids(str(state))
        q_segs = [self._question_segments(q) for q in qs]
        fixed = len(self._prefix) + len(self._suffix) + len(head) + sum(len(s) for segs, _, _ in q_segs for s in segs)
        budget = self.max_length - fixed
        if budget < 1:
            raise ValueError(
                f"The questions alone need {fixed} tokens but max_length={self.max_length}; "
                "raise max_length or shorten the options."
            )
        state_ids = state_ids[:budget]

        ids = self._prefix + head + state_ids
        q_pos, opt_pos, labels = [], [], []
        for q, (segs, opt_idx, ans_idx) in zip(qs, q_segs):
            for i, seg in enumerate(segs):
                ids = ids + seg
                if i in opt_idx and q.kind != NOUL:
                    opt_pos.append(len(ids) - 1)
            q_pos.append(len(ids) - 1)  # last token of the "Answer (...):" segment
            labels.append(parse_label(q, gold.get(q.name)))
        ids = ids + self._suffix

        feats = {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "q_type": [q.kind for q in qs],
            "q_pos": q_pos,
            "q_nopt": [q.num_logits for q in qs],
            "opt_pos": opt_pos,
            "labels": labels,
        }
        return feats, qs

    def __call__(self, example: dict | str, questions: Any = None, gold: Any = None) -> dict:
        if isinstance(example, Mapping):  # datasets.map usage (dict or LazyRow)
            state, questions, gold = example["state"], example["questions"], example.get("gold")
            if isinstance(state, (list, tuple)):  # batched map
                rows = [
                    self.build(s, q, g)[0]
                    for s, q, g in zip(state, questions, gold if gold is not None else [None] * len(state))
                ]
                return {k: [r[k] for r in rows] for k in rows[0]}
            return self.build(state, questions, gold)[0]
        return self.build(example, questions, gold)[0]


# --------------------------------------------------------------------------- collation
@dataclass
class DataCollatorForDecision:
    """Pads encoded examples into a batch (the ``DataCollatorWithPadding`` of decision models)."""

    tokenizer: Any = None
    pad_token_id: int | None = None
    pad_to_multiple_of: int | None = None

    def _pad_id(self) -> int:
        if self.pad_token_id is not None:
            return self.pad_token_id
        tok = self.tokenizer
        for attr in ("pad_token_id", "eos_token_id"):
            value = getattr(tok, attr, None)
            if value is not None:
                return value
        return 0

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        bsz = len(features)
        length = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            length = ((length + m - 1) // m) * m
        n_q = max(len(f["q_type"]) for f in features)
        k = max(2, max(max(f["q_nopt"]) for f in features))

        input_ids = torch.full((bsz, length), self._pad_id(), dtype=torch.long)
        attention_mask = torch.zeros((bsz, length), dtype=torch.long)
        q_type = torch.zeros((bsz, n_q), dtype=torch.long)
        q_pos = torch.zeros((bsz, n_q), dtype=torch.long)
        q_nopt = torch.zeros((bsz, n_q), dtype=torch.long)
        labels = torch.full((bsz, n_q), IGNORE_INDEX, dtype=torch.long)
        opt_pos = torch.zeros((bsz, n_q, k), dtype=torch.long)

        for b, f in enumerate(features):
            n = len(f["input_ids"])
            input_ids[b, :n] = torch.as_tensor(f["input_ids"], dtype=torch.long)
            attention_mask[b, :n] = 1
            nq = len(f["q_type"])
            q_type[b, :nq] = torch.as_tensor(f["q_type"], dtype=torch.long)
            q_pos[b, :nq] = torch.as_tensor(f["q_pos"], dtype=torch.long)
            q_nopt[b, :nq] = torch.as_tensor(f["q_nopt"], dtype=torch.long)
            if "labels" in f and f["labels"] is not None:
                labels[b, :nq] = torch.as_tensor(f["labels"], dtype=torch.long)
            cursor = 0
            flat = f["opt_pos"]
            for qi in range(nq):
                if f["q_type"][qi] == NOUL:
                    continue
                cnt = f["q_nopt"][qi]
                opt_pos[b, qi, :cnt] = torch.as_tensor(flat[cursor : cursor + cnt], dtype=torch.long)
                cursor += cnt
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "q_type": q_type,
            "q_pos": q_pos,
            "q_nopt": q_nopt,
            "opt_pos": opt_pos,
            "labels": labels,
        }
