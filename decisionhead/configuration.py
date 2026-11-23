"""Configuration for decision models."""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig

DEFAULT_TEMPERATURES = {"choice": 1.0, "noul": 1.0, "score": 1.0}


class DecisionConfig(PretrainedConfig):
    """Config of a :class:`~decisionhead.DecisionModel`.

    It wraps the config of an arbitrary backbone (``backbone_config``) and stores
    everything needed to rebuild the decision heads and any LoRA adapters, plus the
    per-question-type calibration temperatures.
    """

    model_type = "decision_model"

    def __init__(
        self,
        backbone_config: dict | PretrainedConfig | None = None,
        hidden_size: int | None = None,
        lora: dict | None = None,
        temperatures: dict | None = None,
        max_input_length: int = 2048,
        head_dim: int = 128,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if backbone_config is not None and not isinstance(backbone_config, dict):
            backbone_config = backbone_config.to_dict()
        self.backbone_config = backbone_config or {}
        if hidden_size is None and self.backbone_config:
            hidden_size = _infer_hidden_size(self.get_backbone_config())
        self.hidden_size = hidden_size
        self.lora = lora
        self.temperatures = {**DEFAULT_TEMPERATURES, **(temperatures or {})}
        self.max_input_length = max_input_length
        self.head_dim = head_dim
        self.initializer_range = initializer_range

    def get_backbone_config(self) -> PretrainedConfig:
        d = dict(self.backbone_config)
        model_type = d.pop("model_type")
        return AutoConfig.for_model(model_type, **d)


def _infer_hidden_size(cfg: PretrainedConfig) -> int:
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    for attr in ("hidden_size", "d_model", "n_embd", "dim"):
        value = getattr(text_cfg, attr, None)
        if isinstance(value, int):
            return value
    raise ValueError(f"Could not infer the hidden size of backbone config {type(cfg).__name__}.")
