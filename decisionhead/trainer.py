"""Trainer + calibration for decision models."""

from __future__ import annotations

import torch
from transformers import Trainer

from .data import DataCollatorForDecision
from .metrics import Record, compute_metrics as default_compute_metrics, fit_temperatures, summarize, unpack_logits


@torch.no_grad()
def collect_records(model, dataset, data_collator=None, batch_size: int = 16) -> list[Record]:
    """Run the model over an encoded dataset and return the raw logits of every labelled question."""
    data_collator = data_collator or DataCollatorForDecision()
    was_training = model.training
    model.eval()
    records: list[Record] = []
    for start in range(0, len(dataset), batch_size):
        batch = data_collator([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))])
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch)
        records += [r for ex in unpack_logits(out.logits.cpu().numpy()) for r in ex]
    model.train(was_training)
    return records


def calibrate(model, dataset, data_collator=None, batch_size: int = 16, update: bool = True) -> dict:
    """Fit one temperature per question type on ``dataset`` (use held-out data!).

    Returns metrics before/after.  With ``update=True`` the temperatures are stored in
    ``model.config.temperatures`` (so they are saved with the model and used by pipelines).
    """
    records = collect_records(model, dataset, data_collator, batch_size)
    temps = fit_temperatures(records)
    before = summarize(records)
    after = summarize(records, temps)
    if update:
        model.config.temperatures = {**model.config.temperatures, **temps}
    return {"temperatures": temps, "before": before, "after": after}


class DecisionTrainer(Trainer):
    """``transformers.Trainer`` with decision-model defaults.

    Defaults ``data_collator`` to :class:`DataCollatorForDecision` and ``compute_metrics`` to
    accuracy / NLL / ECE; everything else (``TrainingArguments``, callbacks, DDP, mixed precision,
    ``load_best_model_at_end``, ``push_to_hub`` ...) is inherited unchanged.
    """

    HEAD_PREFIXES = ("choice_head", "score_head", "noul_head")

    def __init__(self, *args, head_learning_rate: float | None = None, **kwargs):
        # The read-out heads are freshly initialised while the backbone is (usually) pretrained, so a
        # fine-tuning-sized learning rate (e.g. 2e-5) can make the heads learn very slowly.
        # `head_learning_rate` trains them at their own rate (e.g. 1e-3) and the schedule still applies to both.
        self.head_learning_rate = head_learning_rate
        if kwargs.get("data_collator") is None:
            kwargs["data_collator"] = DataCollatorForDecision(tokenizer=kwargs.get("processing_class"))
        if kwargs.get("compute_metrics") is None:
            kwargs["compute_metrics"] = default_compute_metrics
        super().__init__(*args, **kwargs)

    def calibrate(self, eval_dataset=None, update: bool = True) -> dict:
        """Temperature-scale the (best) model on ``eval_dataset`` and return before/after metrics."""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("calibrate() needs an eval_dataset.")
        batch_size = self.args.per_device_eval_batch_size
        return calibrate(self.model, eval_dataset, self.data_collator, batch_size, update)

    def create_optimizer(self, model=None):
        fresh = self.optimizer is None
        optimizer = super().create_optimizer(model)
        if fresh and self.head_learning_rate is not None:
            head_ids = {
                id(p)
                for n, p in (self.model if model is None else model).named_parameters()
                if any(prefix in n for prefix in self.HEAD_PREFIXES)
            }
            for group in list(optimizer.param_groups):
                head = [p for p in group["params"] if id(p) in head_ids]
                if not head:
                    continue
                group["params"] = [p for p in group["params"] if id(p) not in head_ids]
                optimizer.add_param_group({**{k: v for k, v in group.items() if k != "params"}, "params": head, "lr": self.head_learning_rate})
            optimizer.param_groups[:] = [g for g in optimizer.param_groups if g["params"]]
        return optimizer
