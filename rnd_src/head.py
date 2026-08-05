"""Convex linear head used as both the AL model and the A1 proxy environment.

BCE + L2 on frozen features is strictly convex, so LBFGS reaches the same optimum
regardless of initialisation. That determinism is what makes the A1 ground truth
(retrain gain per candidate) a property of the data rather than of the optimiser seed.
"""
import torch
import torch.nn.functional as F


def fit(Z, Y, wd=1e-4, iters=200, W0=None):
    """Fit [W|b] jointly; Z is already augmented so the bias is column D-1."""
    C, D = Y.shape[1], Z.shape[1]
    W = (torch.zeros(C, D, dtype=Z.dtype) if W0 is None else W0.clone())
    W.requires_grad_(True)
    opt = torch.optim.LBFGS([W], max_iter=iters, history_size=10,
                            tolerance_grad=1e-12, tolerance_change=1e-14,
                            line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        # L2 uses mean, not sum: with C*D ~ 41k parameters a summed penalty
        # dominates the mean BCE and pins W near zero (macro-AUPRC collapses to chance).
        loss = F.binary_cross_entropy_with_logits(Z @ W.T, Y) + wd * (W ** 2).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach()


def probs(Z, W):
    return torch.sigmoid(Z @ W.T)


def bce(Z, Y, W):
    return float(F.binary_cross_entropy_with_logits(Z @ W.T, Y))


def fisher_damping(Z, wd):
    """Damping delta consistent with the fitted head's L2 term.

    fit() minimises mean-BCE (averaged over N*C entries) + wd*mean(W^2), whose
    Hessian in block c is (1/(N*C)) Z^T S_c Z + (2*wd/(C*D)) I. ResidualNeed builds
    unnormalised Z^T S_c Z blocks, i.e. the Hessian scaled by N*C, so the matching
    damping is delta = 2*wd*N/D.
    """
    return 2.0 * wd * Z.shape[0] / Z.shape[1]
