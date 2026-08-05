"""Deflation must be exact: Phi_b - Phi_{b+1} == Delta_b(x) from eq.(8).

Also checks that Sherman-Morrison deflation agrees with a from-scratch refactorization,
since the fast path no longer recomputes a Cholesky per pick.
"""
import torch
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
from rnd.scoring import ResidualNeed, augment, reference_gradient, sigma2

N, C, d = 40, 6, 12
H = torch.nn.functional.normalize(torch.randn(N, d), dim=-1)
Z = augment(H)
P = torch.rand(N, C) * 0.8 + 0.1
Y = (torch.rand(N, C) > 0.7).double()
GV = reference_gradient(Z, P, Y)

rn = ResidualNeed(Z, P, GV, delta=0.5)
Zc = augment(torch.nn.functional.normalize(torch.randn(15, d), dim=-1))
Pc = torch.rand(15, C) * 0.8 + 0.1

ok = True
phi = rn.phi()
print('Phi_0 = %.8f' % phi)
for step in range(5):
    sc = rn.score(Zc, Pc, exact=True)
    i = int(sc.argmax())
    pred = float(sc[i])
    rn.deflate(Zc[i], Pc[i])
    new = rn.phi()
    drop = float(phi - new)
    match = abs(drop - pred) < 1e-9
    ok &= match
    print('step%d  pred=%.8f  actual_drop=%.8f  match=%s' % (step, pred, drop, match))
    phi = new
print('EXACT DEFLATION:', ok)

# eq.(9) must upper bound eq.(8) everywhere
rn2 = ResidualNeed(Z, P, GV, delta=0.5)
ex = rn2.score(Zc, Pc, exact=True)
ub = rn2.score(Zc, Pc, exact=False)
print('eq9 >= eq8 for all       :', bool((ub >= ex - 1e-12).all()))
print('mean gap (saturation)    : %.4f' % float((ub - ex).mean()))
print('rank corr eq8 vs eq9     : %.4f' % float(torch.corrcoef(torch.stack([
    ex.argsort().argsort().double(), ub.argsort().argsort().double()]))[0,1]))

# proposition 1: expected linear influence is identically zero under y~p
print('linear influence (y~p)   :', float(rn2.linear_influence(Zc, Pc).abs().max()))


# Sherman-Morrison must match a from-scratch rebuild of A^{-1}.
rn3 = ResidualNeed(Z, P, GV, delta=0.5)
Aref = rn3.A.clone()
for step in range(5):
    z, p = Zc[step], Pc[step]
    rn3.deflate(z, p)
    Aref = Aref + sigma2(p).view(-1, 1, 1) * torch.outer(z, z)
fresh = torch.linalg.solve(Aref, rn3.GV_eff.unsqueeze(-1)).squeeze(-1)
print('SM vs refactorize, max rel err: %.3e' % float(
    ((rn3.R - fresh).norm(dim=-1) / fresh.norm(dim=-1)).max()))
print('A tracked correctly            :', bool(torch.allclose(rn3.A, Aref, atol=1e-9)))
