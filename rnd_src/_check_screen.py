"""How loose is the screen? Measures where the true exact argmax sits in the
upper-bound ordering, which is what determines whether topk-shortlisting is safe."""
import torch
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
from rnd_src.scoring import ResidualNeed, augment, reference_gradient

C, d, NL, NU = 54, 768, 200, 1000
Zl = augment(torch.randn(NL, d) * 0.3)
Pl = torch.rand(NL, C) * 0.8 + 0.1
Y = (torch.rand(NL, C) > 0.8).double()
GV = reference_gradient(Zl, Pl, Y)
Zu = augment(torch.randn(NU, d) * 0.3)
Pu = torch.rand(NU, C) * 0.8 + 0.1

rn = ResidualNeed(Zl, Pl, GV, delta=1.0)
ex = rn.score_exact(Zu, Pu)
for name, ub in (('eq.(9) score_lin', rn.score_lin(Zu, Pu)),
                 ('score_ub (trace)', rn.score_ub(Zu, Pu))):
    order = ub.argsort(descending=True)
    true_best = int(ex.argmax())
    rank = int((order == true_best).nonzero()[0, 0])
    ratio = float((ub / ex).max())
    # does a topk=K shortlist contain the true argmax?
    hits = {K: bool((order[:K] == true_best).any()) for K in (1, 8, 32, 128)}
    print('%-18s max ub/exact=%.2f  rank of true argmax=%d  topK hit=%s'
          % (name, ratio, rank, hits))
