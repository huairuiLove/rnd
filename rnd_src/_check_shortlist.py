"""Audit: does the topk shortlist recover the exact argmax at every greedy step?

The eq.(9)-style certificate in select() is sound but loose (ub/exact ~ 14x), so it
almost never fires. That does NOT mean the shortlist is wrong -- it means the bound is
too weak to prove it. This measures the thing that actually matters by computing the
exact score over the whole work set at every step and comparing.
"""
import torch
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
from rnd_src.scoring import ResidualNeed, augment, reference_gradient

C, d, NL, M, B = 54, 768, 200, 400, 30
Zl = augment(torch.randn(NL, d) * 0.3)
Pl = torch.rand(NL, C) * 0.8 + 0.1
Y = (torch.rand(NL, C) > 0.8).double()
GV = reference_gradient(Zl, Pl, Y)
Zu = augment(torch.randn(M, d) * 0.3)
Pu = torch.rand(M, C) * 0.8 + 0.1

for topk in (8, 32, 128):
    rn = ResidualNeed(Zl, Pl, GV, delta=1.0)
    alive = torch.ones(M, dtype=torch.bool)
    agree = 0
    lost = 0.0
    for b in range(B):
        ub = rn.score_ub(Zu, Pu); ub[~alive] = -float('inf')
        full = rn.score_exact(Zu, Pu); full[~alive] = -float('inf')
        cand = ub.topk(min(topk, int(alive.sum()))).indices
        i_short = int(cand[int(rn.score_exact(Zu[cand], Pu[cand]).argmax())])
        i_true = int(full.argmax())
        agree += (i_short == i_true)
        lost += float(full[i_true] - full[i_short]) / max(float(full[i_true]), 1e-300)
        rn.deflate(Zu[i_short], Pu[i_short]); alive[i_short] = False
    print('topk=%-4d shortlist==full argmax: %2d/%d steps   mean rel gain lost: %.3e'
          % (topk, agree, B, lost / B))
