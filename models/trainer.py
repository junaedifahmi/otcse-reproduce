from typing import Tuple

import torch
import torch.nn.functional as F

from .optimal_transport import OTModule
from .simsiam import SimSiam


class OTCSETrainer:
    def __init__(self, model, dataset, trainer_args):

        # model should be modular
        self.model = model
        self.ssl_module = SimSiam(model)

        # settings for optimal transport
        # self.alpha = 0.1  # scaling cost matrix for Cost Feature
        # self.beta = 0.001  # scaling cost matrix for Class Feature
        # self.reg = 0.5  # regularization strength 1/sigma

        # self.epsilon = 0.01
        # self.scale = 5.0

        self.ot_module = OTModule(trainer_args.optimal_transport)
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # trainer arguments for OTCSE (this class)
        self.mu = 0.7  # for jot loss scaling
        self.nu = 0.7  # for ssl loss scaling

    def train(self):
        for epoch in range(2):  # iter over epochs
            for d_s, d_t in self.dataset:  # iter over batches
                loss = self._train_batch(d_s, d_t)
                print(f"Epoch {epoch}, Loss: {loss}")

    def _train_batch(self, d_s: Tuple[torch.Tensor, torch.Tensor], d_t: torch.Tensor):
        x_s, labels_s = d_s
        x_t = d_t

        # forward pass on the source domain
        e_s, logits_s = self.model(x_s)

        # forward pass on the target domain
        e_t, logits_t = self.model(x_t)

        # implement sinkhorn OT optimization on C adapt to get gamma*
        probs_s = logits_s.softmax(dim=1)
        probs_t = logits_t.softmax(dim=1)

        loss_jot = self.ot_module.compute_loss(e_s, e_t, probs_s, probs_t)
        loss_ssl = self.ssl_module.compute_loss(x_t)
        loss_ce = F.cross_entropy(logits_s, labels_s)  # loss for classifier

        # Total Loss
        loss_total = loss_ce + (self.mu * loss_jot) + (self.nu * loss_ssl)

        # Optimize the model
        self.optimizer.zero_grad()
        loss_total.backward()

        self.optimizer.step()

        return loss_total
