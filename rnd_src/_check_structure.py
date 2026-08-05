"""Sanity checks for the structural claims the RND derivation depends on."""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
C, d, n = 5, 7, 3
lin = nn.Linear(d, C)
h = torch.randn(n, d)
y = (torch.rand(n, C) > 0.5).float()

logits = lin(h)
loss = F.binary_cross_entropy_with_logits(logits, y, reduction='sum')
gW, gb = torch.autograd.grad(loss, [lin.weight, lin.bias])
p = torch.sigmoid(logits)

print('grad_W == (p-y)^T h      :', torch.allclose(gW, (p - y).T @ h, atol=1e-5))
print('grad_b == sum(p-y)       :', torch.allclose(gb, (p - y).sum(0), atol=1e-5))

ha = torch.cat([h, torch.ones(n, 1)], 1)
G = (p - y).T @ ha
print('augmented [W|b] matches  :', torch.allclose(G, torch.cat([gW, gb[:, None]], 1), atol=1e-5))
ranks = [int(torch.linalg.matrix_rank(torch.outer((p - y)[i], ha[i]))) for i in range(n)]
print('per-sample rank (aug)    :', ranks)

# psi_c orthonormality requires ||h||=1; check what breaks without normalisation
hn = F.normalize(ha, dim=-1)
print('||h_aug|| before norm    :', [round(v, 3) for v in ha.norm(dim=-1).tolist()])
print('||h_aug|| after norm     :', [round(v, 3) for v in hn.norm(dim=-1).tolist()])

# Woodbury identity behind eq.(8): exact marginal gain
delta = 0.7
Cd = C * (d + 1)
A = delta * torch.eye(Cd)
gV = torch.randn(Cd)
x = hn[0]
s2 = (p[0] * (1 - p[0]))
Psi = torch.zeros(Cd, C)
for c in range(C):
    Psi[c * (d + 1):(c + 1) * (d + 1), c] = x
S = torch.diag(s2)
A1 = A + Psi @ S @ Psi.T
phi0 = gV @ torch.linalg.solve(A, gV)
phi1 = gV @ torch.linalg.solve(A1, gV)
r = torch.linalg.solve(A, gV)
v = Psi.T @ r
M = Psi.T @ torch.linalg.solve(A, Psi)
exact = v @ torch.linalg.solve(torch.linalg.inv(S) + M, v)
print('eq.(8) exact gain        :', torch.allclose(phi0 - phi1, exact, atol=1e-5))
lin_ub = (s2 * v ** 2).sum()
print('eq.(9) upper bound holds :', bool(lin_ub >= exact - 1e-6),
      'exact=%.5f ub=%.5f' % (exact, lin_ub))
