"""decisionhead: fine-tune any model into a Jev-style (System One) decision model.

The API mirrors Hugging Face ``transformers``::

    from decisionhead import AutoTokenizer, AutoModelForDecision, DecisionEncoder
    from decisionhead import DataCollatorForDecision, TrainingArguments, Trainer, pipeline
"""

from transformers import AutoTokenizer, TrainingArguments

from .configuration import DecisionConfig
from .data import (
    Choice,
    DataCollatorForDecision,
    DecisionEncoder,
    Noul,
    Score,
    make_example,
    parse_questions,
)
from .metrics import compute_metrics, unpack_logits
from .modeling import AutoModelForDecision, DecisionModel
from .pipelines import DecisionPipeline, pipeline
from .trainer import DecisionTrainer, calibrate

Trainer = DecisionTrainer  # so `from decisionhead import Trainer` just works

__version__ = "0.1.0"

__all__ = [
    "AutoModelForDecision",
    "AutoTokenizer",
    "Choice",
    "DataCollatorForDecision",
    "DecisionConfig",
    "DecisionEncoder",
    "DecisionModel",
    "DecisionPipeline",
    "DecisionTrainer",
    "Noul",
    "Score",
    "Trainer",
    "TrainingArguments",
    "calibrate",
    "compute_metrics",
    "make_example",
    "parse_questions",
    "pipeline",
    "unpack_logits",
]
