"""Offline helpers: a tiny random backbone + tokenizer and a synthetic ticket-triage dataset.

Handy for smoke-testing the training loop without downloading anything::

    from decisionhead.testing import make_tiny_backbone, make_synthetic_dataset
    path = make_tiny_backbone("/tmp/tiny-base")
"""

from __future__ import annotations

import random

TEAMS = {
    "billing": ["invoice", "charge", "refund", "payment", "card"],
    "technical": ["error", "crash", "bug", "outage", "login"],
    "sales": ["price", "plan", "quote", "upgrade", "demo"],
}
TEAM_DESCRIPTIONS = {
    "billing": "invoices payments refunds",
    "technical": "bugs outages errors",
    "sales": "pricing new plans",
}
URGENT = ["asap", "urgent", "immediately"]
FILLER = ["please", "help", "my", "the", "is", "again", "today", "thanks", "with", "about", "our", "hello", "need", "we"]

QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": TEAM_DESCRIPTIONS},
    "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
    "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]},
}


def make_synthetic_rows(n: int, seed: int = 0) -> list[dict]:
    """Rows with ``state`` / ``questions`` / ``gold`` whose answers follow simple keyword rules."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        team = rng.choice(list(TEAMS))
        words = [rng.choice(TEAMS[team])] + [rng.choice(FILLER) for _ in range(rng.randint(3, 7))]
        n_urgent = rng.choice([0, 0, 1, 2])
        words += rng.sample(URGENT, n_urgent)
        if rng.random() < 0.3:
            words.append("refund")
        rng.shuffle(words)
        text = " ".join(words)
        rows.append(
            {
                "state": text,
                "questions": QUESTIONS,
                "gold": {"team": team, "refund": "refund" in words, "urgency": n_urgent},
            }
        )
    return rows


def make_synthetic_dataset(n: int, seed: int = 0):
    """Same as :func:`make_synthetic_rows` but as a ``datasets.Dataset`` (JSON-string columns)."""
    from datasets import Dataset

    from .data import make_example

    return Dataset.from_list([make_example(r["state"], r["questions"], r["gold"]) for r in make_synthetic_rows(n, seed)])


def make_tiny_backbone(path: str, hidden_size: int = 64, layers: int = 2, vocab_size: int = 400, seed: int = 0) -> str:
    """Save a randomly initialised tiny Qwen2 model plus a BPE tokenizer to ``path`` and return it."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2Model

    corpus = []
    for row in make_synthetic_rows(300, seed=123):
        corpus.append(row["state"])
    corpus += [
        "State: Question (team): Which team should handle this? Options: - billing: invoices payments refunds",
        "Question (refund): Does the customer ask for a refund? Answer (refund): Levels, lowest to highest: low medium high",
        "- technical: bugs outages errors - sales: pricing new plans Question (urgency): How urgent is this? Answer (team):",
    ] * 20
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<unk>", "<eos>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>")
    fast.save_pretrained(path)

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=len(fast),
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=fast.pad_token_id,
        tie_word_embeddings=False,
    )
    Qwen2Model(cfg).save_pretrained(path)
    return path
