"""Cost check at CoMAL's real shape: C=54, d=769, budget=100, pool=20k.

The earlier implementation refactorized 54 769x769 Choleskys per pick (~2.5e12 flops
for a budget of 100) and exact-scored the whole pool before truncating. This measures
what the screen + Sherman-Morrison path actually costs.
"""
import time
import torch
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
from rnd_src.scoring import ResidualNeed, augment, reference_gradient, select

C, d, NL, NU, B = 54, 768, 200, 20000, 100
Zl = augment(torch.randn(NL, d) * 0.3)
Pl = torch.rand(NL, C) * 0.8 + 0.1
Y = (torch.rand(NL, C) > 0.8).double()
GV = reference_gradient(Zl, Pl, Y)
Zu = augment(torch.randn(NU, d) * 0.3)
Pu = torch.rand(NU, C) * 0.8 + 0.1

t = time.time(); rn = ResidualNeed(Zl, Pl, GV, delta=1.0); t_build = time.time() - t
print('build A + invert (C=%d, D=%d): %.2fs' % (C, d + 1, t_build))

t = time.time(); ub = rn.score_ub(Zu, Pu); t_screen = time.time() - t
print('screen over %d pool           : %.2fs' % (NU, t_screen))
# both bounds must dominate the exact score on a sample, else the screen is invalid
sub = torch.randperm(NU)[:300]
ex_s = rn.score_exact(Zu[sub], Pu[sub])
print('score_ub  >= exact everywhere  :', bool((rn.score_ub(Zu[sub], Pu[sub]) >= ex_s - 1e-12).all()))
print('score_lin >= score_ub          :', bool((rn.score_lin(Zu[sub], Pu[sub]) >= rn.score_ub(Zu[sub], Pu[sub]) - 1e-12).all()))

work = 10 * B
top = ub.topk(work).indices
t = time.time()
picks, trace = select(rn, Zu[top], Pu[top], B, exact=True)
t_sel = time.time() - t
print('greedy %d picks over %d work  : %.2fs' % (B, work, t_sel))
print('TOTAL                          : %.2fs' % (t_build + t_screen + t_sel))
print('cached-m drift (max rel err)    : %.3e' % trace[0]['m_drift'])
print('max rel err (pred vs realised) : %.3e' % max(x['rel_err'] for x in trace))
phis = [trace[0]['phi_before']] + [x['phi_after'] for x in trace]
print('Phi monotone                   :', all(b >= a - 1e-12 for a, b in zip(phis[1:], phis[:-1])))
print('Phi %.4e -> %.4e (drop %.1f%%)' % (phis[0], phis[-1], 100*(phis[0]-phis[-1])/phis[0]))
