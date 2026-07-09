from torch import nn
from torch.nn import functional as F


class SimSiam(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.predictor = nn.Linear(
            model.embedding_dim,
            model.embedding_dim,
        )

    def augment(self, x):
        x1, x2 = x, x
        return x1, x2

    def forward(self, x1, x2):
        z1 = self.model.encode(x1)
        z2 = self.model.encode(x2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        loss1 = -F.cosine_similarity(
            p1,
            z2.detach(),
            dim=1,
        ).mean()

        loss2 = -F.cosine_similarity(
            p2,
            z1.detach(),
            dim=1,
        ).mean()

        return (loss1 + loss2) / 2

    def compute_loss(self, x):
        x1, x2 = self.augment(x)
        return self.forward(x1, x2)
