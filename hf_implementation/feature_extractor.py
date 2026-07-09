"""
Unified interface over pretrained self-supervised speech backbones.

`dataset.py` only talks to `AudioFeatureExtractor` -- it doesn't need to
know whether HuBERT, Wav2Vec2, or something else is doing the work.

Usage:
    extractor = AudioFeatureExtractor(backend="hubert")
    embedding = extractor(waveform, sampling_rate=16000)  # -> (hidden_size,)
"""

from typing import Optional

import torch
from transformers import (
    HubertModel,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)

# backend name -> (default HF checkpoint, model class)
_BACKENDS = {
    "hubert": ("facebook/hubert-base-ls960", HubertModel),
    "wav2vec2": ("facebook/wav2vec2-base-960h", Wav2Vec2Model),
}


class AudioFeatureExtractor:
    """Frozen self-supervised speech feature extractor.

    Args:
        backend: one of "hubert", "wav2vec2".
        checkpoint: HF checkpoint to load; defaults to the standard base
            checkpoint for the chosen backend if not given.
        pooling: "mean" or "max" pooling of the model's last hidden state
            over time, producing a single fixed-size vector per utterance.
        sampling_rate: expected input sampling rate (must match how the
            audio was resampled upstream, e.g. in `build_emodb_datasets`).
        device: device to run the backbone on. Defaults to "cpu" -- this
            runs inside `Dataset.__getitem__`, which (with num_workers > 0)
            executes in separate worker processes; CUDA/MPS tensors don't
            reliably share across those, so CPU is the safe default here.
            The pooled output is always moved back to CPU before being
            returned; `Trainer` moves the whole batch to the training
            device (e.g. mps) right before it reaches your model, so this
            extractor's device is independent of that.
    """

    def __init__(
        self,
        backend: str = "hubert",
        checkpoint: Optional[str] = None,
        pooling: str = "mean",
        sampling_rate: int = 16000,
        device: str = "cpu",
    ):
        if backend not in _BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose from {list(_BACKENDS)}"
            )
        if pooling not in ("mean", "max"):
            raise ValueError(f"Unknown pooling '{pooling}'. Choose 'mean' or 'max'")

        default_checkpoint, model_cls = _BACKENDS[backend]
        checkpoint = checkpoint or default_checkpoint

        self.backend = backend
        self.pooling = pooling
        self.sampling_rate = sampling_rate
        self.device = torch.device(device)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint)
        self.model = model_cls.from_pretrained(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def output_dim(self) -> int:
        return self.model.config.hidden_size

    def __call__(self, waveform: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        inputs = self.feature_extractor(
            waveform.numpy(), sampling_rate=sampling_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden_states = self.model(
                **inputs
            ).last_hidden_state  # (1, T, hidden_size)
        pooled = (
            hidden_states.mean(dim=1)
            if self.pooling == "mean"
            else hidden_states.max(dim=1).values
        )
        return pooled.squeeze(
            0
        ).cpu()  # always return CPU tensors -- Trainer moves the batch later
