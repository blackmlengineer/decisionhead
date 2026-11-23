"""Decision model: any backbone + read-out heads that answer typed questions in ONE forward pass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.utils import ModelOutput

from .configuration import DecisionConfig
from .data import CHOICE, IGNORE_INDEX, NOUL, SCORE

NEG = -1e9
_HUB_KWARGS = ("revision", "token", "cache_dir", "trust_remote_code", "local_files_only", "subfolder", "force_download")


@dataclass
class DecisionOutput(ModelOutput):
    """``logits`` is a packed 2-D tensor (see :func:`decisionhead.metrics.unpack_logits`).

    Packing keeps the output compatible with ``transformers.Trainer`` evaluation loops, which
    concatenate predictions across batches that may have different numbers of questions/options.
    """

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


def normalize_lora(lora) -> dict | None:
    """Accept ``None/False``, ``True`` (sane defaults), a dict, or a ``peft.LoraConfig``."""
    if lora is None or lora is False:
        return None
    default = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": "all-linear"}
    if lora is True:
        return default
    if isinstance(lora, dict):
        return {**default, **lora}
    # peft.LoraConfig
    tm = lora.target_modules
    return {
        "r": lora.r,
        "lora_alpha": lora.lora_alpha,
        "lora_dropout": lora.lora_dropout,
        "target_modules": sorted(tm) if isinstance(tm, (set, list, tuple)) else tm,
        "use_rslora": getattr(lora, "use_rslora", False),
    }


class PointerHead(nn.Module):
    """Scores options with ``query(answer position) . key(option) / sqrt(d) + linear(option)``.

    The hidden state at the end of the question has seen the state *and* every option, so it can
    "point" at the matching option; the linear term adds a per-option prior.
    """

    def __init__(self, hidden: int, head_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden, head_dim)
        self.key = nn.Linear(hidden, head_dim)
        self.bias = nn.Linear(hidden, 1)
        self.scale = head_dim**-0.5
        for m in (self.query, self.key, self.bias):
            m._decision_head = True

    def forward(self, h_opt: torch.Tensor, h_q: torch.Tensor) -> torch.Tensor:
        # h_opt: [B, Q, K, H], h_q: [B, Q, H] -> [B, Q, K]
        sim = torch.einsum("bqd,bqkd->bqk", self.query(h_q), self.key(h_opt)) * self.scale
        return sim + self.bias(h_opt).squeeze(-1)


class DecisionModel(PreTrainedModel):
    """Backbone + three tiny linear heads.

    * ``choice`` / ``score``: a pointer read-out. The hidden state at the end of the question
      (the query) is matched against the hidden state at the last token of every option (the
      keys); a softmax over the question's options gives the probabilities.
    * ``noul``: the hidden state at the end of the question maps to a yes/no logit.

    All questions of an example share one sequence, so they are answered in a single pass.
    """

    config_class = DecisionConfig
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def __init__(self, config: DecisionConfig, backbone: nn.Module | None = None):
        super().__init__(config)
        if backbone is None:
            backbone = AutoModel.from_config(config.get_backbone_config())
        self.backbone = backbone
        hidden = config.hidden_size
        head_dim = min(hidden, config.head_dim)
        self.choice_head = PointerHead(hidden, head_dim)
        self.score_head = PointerHead(hidden, head_dim)
        self.noul_head = nn.Linear(hidden, 1)
        self.noul_head._decision_head = True
        self.post_init()
        if config.lora:  # re-create adapters when loading a saved LoRA model
            self._inject_lora(config.lora)

    # ------------------------------------------------------------------ init / embeddings
    def _init_weights(self, module):
        # Only touch our heads; the backbone owns (and has already loaded) its own weights.
        if getattr(module, "_decision_head", False):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            nn.init.zeros_(module.bias)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        if hasattr(self.backbone, "activation_checkpointing_enable"):  # transformers >= 5.x name
            self.backbone.activation_checkpointing_enable(activation_checkpointing_kwargs=gradient_checkpointing_kwargs, **kwargs)
        else:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        if self.config.lora:  # frozen embeddings need grads on their outputs for checkpointing
            self.backbone.enable_input_require_grads()

    def gradient_checkpointing_disable(self):
        if hasattr(self.backbone, "activation_checkpointing_disable"):
            self.backbone.activation_checkpointing_disable()
        else:
            self.backbone.gradient_checkpointing_disable()

    # ------------------------------------------------------------------ LoRA
    def add_lora(self, lora=True):
        """Freeze the backbone and train low-rank adapters (+ the heads) instead."""
        if self.config.lora:
            raise ValueError("This model already has LoRA adapters; call merge_lora() first.")
        self.config.lora = normalize_lora(lora)
        self._inject_lora(self.config.lora)
        return self

    def _inject_lora(self, lora_cfg: dict):
        from peft import LoraConfig, inject_adapter_in_model

        inject_adapter_in_model(LoraConfig(**lora_cfg), self.backbone)
        for name, param in self.backbone.named_parameters():
            param.requires_grad = "lora_" in name

    def merge_lora(self):
        """Bake the adapters into the backbone weights and drop them (for export / inference)."""
        from peft.tuners.lora import LoraLayer

        for name, module in list(self.backbone.named_modules()):
            if isinstance(module, LoraLayer):
                module.merge()
                parent_name, _, child = name.rpartition(".")
                parent = self.backbone.get_submodule(parent_name) if parent_name else self.backbone
                setattr(parent, child, module.get_base_layer())
        self.config.lora = None
        return self

    # ------------------------------------------------------------------ forward
    def decision_logits(self, input_ids, attention_mask, q_type, q_pos, q_nopt, opt_pos) -> torch.Tensor:
        """Raw (uncalibrated) logits of shape ``[batch, questions, max_options]``; padding is ``-1e9``."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        bsz, n_q, k = opt_pos.shape
        dim = hidden.size(-1)
        head_dtype = self.noul_head.weight.dtype

        h_opt = hidden.gather(1, opt_pos.reshape(bsz, n_q * k, 1).expand(-1, -1, dim)).view(bsz, n_q, k, dim)
        h_q = hidden.gather(1, q_pos.unsqueeze(-1).expand(-1, -1, dim))
        h_opt, h_q = h_opt.to(head_dtype), h_q.to(head_dtype)

        s_choice = self.choice_head(h_opt, h_q)
        s_score = self.score_head(h_opt, h_q)
        z = self.noul_head(h_q).squeeze(-1)  # logit of "yes"
        s_noul = torch.zeros_like(s_choice)
        s_noul[..., 1] = z  # classes: [no, yes] -> softmax == sigmoid

        t = q_type.unsqueeze(-1)
        logits = torch.where(t == CHOICE, s_choice, torch.where(t == SCORE, s_score, s_noul))
        valid = torch.arange(k, device=logits.device).view(1, 1, k) < q_nopt.unsqueeze(-1)
        return logits.masked_fill(~valid, NEG)

    @staticmethod
    def _pack(logits, q_type, q_nopt, labels) -> torch.Tensor:
        bsz, n_q, k = logits.shape
        meta = torch.stack([q_type.float(), labels.float(), q_nopt.float()], dim=-1)
        body = torch.cat([meta, logits.float()], dim=-1).reshape(bsz, -1)
        header = torch.tensor([n_q, k], dtype=torch.float32, device=logits.device).expand(bsz, 2)
        return torch.cat([header, body], dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        q_type: torch.Tensor | None = None,
        q_pos: torch.Tensor | None = None,
        q_nopt: torch.Tensor | None = None,
        opt_pos: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> DecisionOutput:
        logits = self.decision_logits(input_ids, attention_mask, q_type, q_pos, q_nopt, opt_pos)
        loss = None
        if labels is not None:
            if (labels != IGNORE_INDEX).any():
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX)
            else:
                loss = logits.sum() * 0.0
        else:
            labels = torch.full_like(q_type, IGNORE_INDEX)
        return DecisionOutput(loss=loss, logits=self._pack(logits.detach(), q_type, q_nopt, labels))


class AutoModelForDecision:
    """``AutoModelForSequenceClassification`` for decision models.

    ``from_pretrained`` accepts either

    * any backbone checkpoint (Hub id or local path) -> a fresh decision model on top of it, or
    * a directory saved by ``DecisionModel.save_pretrained`` -> the trained decision model.

    Extra keyword arguments (``dtype``, ``attn_implementation``, ``device_map`` ...) go to the
    backbone's ``from_pretrained``.  ``lora=True`` (or a dict / ``LoraConfig``) trains adapters only.
    """

    def __init__(self, *args, **kwargs):
        raise EnvironmentError("Use AutoModelForDecision.from_pretrained(...).")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, lora=None, max_length: int | None = None, **kwargs):
        hub = {k: kwargs[k] for k in _HUB_KWARGS if k in kwargs}
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **hub)
        if config.model_type == DecisionConfig.model_type:
            model = DecisionModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
            if lora and not model.config.lora:
                model.add_lora(lora)
            return model

        backbone = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        for flag in ("use_cache",):
            if hasattr(backbone.config, flag):
                setattr(backbone.config, flag, False)
        dconfig = DecisionConfig(backbone_config=backbone.config, **({"max_input_length": max_length} if max_length else {}))
        model = DecisionModel(dconfig, backbone=backbone)
        if lora:
            model.add_lora(lora)
        return model

    @classmethod
    def from_config(cls, backbone_config, lora=None, **kwargs):
        """Randomly initialised decision model (handy for tests and from-scratch training)."""
        dconfig = DecisionConfig(backbone_config=backbone_config, **kwargs)
        model = DecisionModel(dconfig)
        if lora:
            model.add_lora(lora)
        return model


try:  # lets AutoConfig.from_pretrained(<saved decision model dir>) work
    AutoConfig.register(DecisionConfig.model_type, DecisionConfig, exist_ok=True)
except TypeError:  # older transformers without exist_ok
    try:
        AutoConfig.register(DecisionConfig.model_type, DecisionConfig)
    except ValueError:
        pass
