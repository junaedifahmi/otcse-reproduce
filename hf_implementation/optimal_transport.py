import ot
import torch


class OTModule:
    def __init__(self, alpha=1.0, beta=1.0, reg=1e-3, scale=1.0, epsilon=1e-3):
        self.alpha = alpha  # for feature scaling
        self.beta = beta  # for class scaling
        self.reg = reg  # regularization sinkhorn
        self.scale = scale  # scale for sigmoid activation
        self.epsilon = epsilon  # threshold for sinkhorn OT

    def _C_feat(self, e_s, e_t):
        # compute the feature distance matrix
        return torch.cdist(e_s, e_t).pow(2)

    def _C_cls(self, y_s, y_t):
        # compute the class distance matrix
        return torch.cdist(y_s, y_t).pow(2)

    def compute_gamma_star(self, e_s, e_t, y_s, y_t, debug=False):
        # implement sinkhorn OT optimization on C adapt to get gamma*
        c_feat = self._C_feat(e_s, e_t)
        c_cls = self._C_cls(y_s, y_t)

        # got Cost Matrix adaptation
        c_adapt = (self.alpha * c_feat) + (self.beta * c_cls)

        # debug
        if debug:
            print("c_feat mean:", c_feat.mean().item())
            print("c_feat max :", c_feat.max().item())
            print("c_cls mean :", c_cls.mean().item())
            print("c_cls max  :", c_cls.max().item())
            print("c_adapt mean:", c_adapt.mean().item())
            print("c_adapt max :", c_adapt.max().item())

        ## Mass distribution (as uniform)
        bs = e_s.size(0)
        bt = e_t.size(0)

        # mass from source distribution
        a = (
            torch.ones(
                bs,
                device=e_s.device,
            )
            / bs
        )

        # mass from target distribution
        b = (
            torch.ones(
                bt,
                device=e_t.device,
            )
            / bt
        )

        # OT optimization with sinkhorn
        gamma_star, _ = ot.bregman.sinkhorn(
            a,
            b,
            c_adapt,
            reg=self.reg,
            log=True,
        )
        return c_adapt, gamma_star

    def compute_loss(self, e_s, e_t, y_s, y_t):
        c_adapt, gamma_star = self.compute_gamma_star(e_s, e_t, y_s, y_t)

        # compute the OT as loss by adding scaling and sigmoid activation
        w = torch.sigmoid(self.scale * (self.epsilon - gamma_star))
        loss_ot = (c_adapt * gamma_star * w).sum()

        return loss_ot
