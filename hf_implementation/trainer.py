"""
OTCSE training on top of Hugging Face `transformers.Trainer`.

Design notes
------------
The original loop trained on paired batches (d_s, d_t) where:
    d_s = (x_s, labels_s)   -> source domain batch
    d_t = x_t               -> target domain batch (unlabeled)

`Trainer` expects a single `Dataset` yielding dict-like samples and a model
called as `model(**inputs)`. To keep the two-domain pairing, we:

  1. Wrap the source/target datasets in `PairedDomainDataset`, which returns
     one dict per sample: {"x_s": ..., "labels_s": ..., "x_t": ...}.
  2. Subclass `Trainer` and override `compute_loss` to reproduce the exact
     original loss computation (CE + mu * JOT + nu * SSL).
  3. Leave optimization, logging, checkpointing, epochs, etc. to `Trainer`
     itself, all driven by `TrainingArguments` built from the YAML config.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from transformers import Trainer

from .optimal_transport import OTModule
from .simsiam import SimSiam


@dataclass
class OTCSEArguments:
    """Loss-weighting / OT hyperparameters (the `otcse` + `optimal_transport`
    sections of the YAML config)."""

    mu: float = 0.7  # scaling for the JOT loss
    nu: float = 0.7  # scaling for the SSL loss

    ot_alpha: float = 0.1
    ot_beta: float = 0.001
    ot_reg: float = 0.5
    ot_epsilon: float = 0.01
    ot_scale: float = 5.0

    def ot_module_config(self) -> Dict[str, float]:
        return {
            "alpha": self.ot_alpha,
            "beta": self.ot_beta,
            "reg": self.ot_reg,
            "epsilon": self.ot_epsilon,
            "scale": self.ot_scale,
        }


class OTCSETrainer(Trainer):
    """`transformers.Trainer` subclass implementing the OTCSE loss.

    `model` must be callable as `model(x) -> (embedding, logits)`, matching
    the original code (`e_s, logits_s = self.model(x_s)`).
    """

    def __init__(
        self,
        *args,
        otcse_args: OTCSEArguments,
        ot_module: Optional[OTModule] = None,
        ssl_module: Optional[SimSiam] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.otcse_args = otcse_args
        self.ot_module = ot_module or OTModule(**otcse_args.ot_module_config())
        # SimSiam wraps the same backbone as `self.model`
        self.ssl_module = ssl_module or SimSiam(self.model)
        self.ssl_module.to(self.args.device)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        x_s = inputs["x_s"]
        labels_s = inputs["labels_s"]
        x_t = inputs["x_t"]

        # forward pass on the source domain
        e_s, logits_s = model(x_s)
        # forward pass on the target domain
        e_t, logits_t = model(x_t)

        probs_s = logits_s.softmax(dim=1)
        probs_t = logits_t.softmax(dim=1)

        loss_jot = self.ot_module.compute_loss(e_s, e_t, probs_s, probs_t)
        loss_ssl = self.ssl_module.compute_loss(x_t)
        loss_ce = F.cross_entropy(logits_s, labels_s)

        loss_total = (
            loss_ce + (self.otcse_args.mu * loss_jot) + (self.otcse_args.nu * loss_ssl)
        )

        if return_outputs:
            return loss_total, {"logits_s": logits_s, "lgits_t": logits_t}
        return loss_total

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Evaluation uses only the classifier head on the source-style
        batch (x_s, labels_s) -- no target-domain / SSL / OT loss involved,
        so eval datasets don't need an `x_t` field."""
        x_s = inputs["x_s"]
        labels_s = inputs["labels_s"]

        with torch.no_grad():
            _, logits_s = model(x_s)
            loss = F.cross_entropy(logits_s, labels_s)

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits_s, labels_s)
