from torch.nn.functional import normalize
from .standization import Standization
from .base import BaseMethod
from torch import empty_like, eye, norm, cat, randperm, trace
from numpy import sqrt

def frobenius_inner_product(A, B):
    return trace(A.T @ B)

def act(x0, x1, lambda_param=1e-3):
    N = x0.size(0)
    D = x0.size(1)
    
    x0 = sqrt(D) * normalize(x0)
    x1 = sqrt(D) * normalize(x1)

    c = x0.T @ x1 / N # DxD
    c_diff = c - eye(D, device=c.device)
    G = c_diff.detach()
    return norm(x0 - x1, p=2, dim=1).pow(2).mean() + lambda_param * frobenius_inner_product(c_diff, G)

class ACT(BaseMethod):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.standization = Standization(cfg.emb, eps=cfg.s_eps, track_running_stats=False)
        self.loss_f = act
        self.s_iter = cfg.s_iter
        self.s_size = cfg.bs if cfg.s_size is None else cfg.s_size

    def forward(self, samples):
        bs = len(samples[0])
        h = [self.model(x.cuda(non_blocking=True)) for x in samples]
        h = self.head(cat(h))
        loss = 0
        for _ in range(self.s_iter):
            z = empty_like(h)
            perm = randperm(bs).view(-1, self.s_size)
            for idx in perm:
                for i in range(len(samples)):
                    z[idx + i * bs] = self.standization(h[idx + i * bs])
            for i in range(len(samples) - 1):
                for j in range(i + 1, len(samples)):
                    x0 = z[i * bs: (i + 1) * bs]
                    x1 = z[j * bs: (j + 1) * bs]
                    loss += self.loss_f(x0, x1)
        loss /= self.s_iter * self.num_pairs
        return loss

