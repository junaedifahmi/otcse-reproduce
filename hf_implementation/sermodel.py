from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers.utils.generic import ModelOutput


@dataclass
class SERModelOutput(ModelOutput):
    embedding: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class SERModel(nn.Module):
    def __init__(
        self,
        input_dim=768,
        embedding_dim=300,
        num_classes=4,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.encoder = nn.Linear(
            input_dim,
            embedding_dim,
        )

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes,
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        e = self.encode(x)
        logits = self.classifier(e)

        return e, logits
