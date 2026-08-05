"""Verify: for a single sigmoid head, F(x) is block-diagonal per label,
so eq.(8) collapses to a per-label scalar form."""
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

C, D, n = 4, 6, 5           # D = d+1 (augmented)
Z = torch.randn(n, D)
P = torch.rand(n, C) * 0.8 + 0.1
delta = 0.3

def psi(z):
    Psi = torch.zeros(C * D, C)
    for c in range(C):
        Psi[c * D:(c + 1) * D, c] = z
    return Psi

# A_0 = delta I + sum_x F(x)
A = delta * torch.eye(C * D)
for i in range(n):
    s2 = P[i] * (1 - P[i])
    Ps = psi(Z[i])
    A += Ps @ torch.diag(s2) @ Ps.T

# block-diagonal claim: A[c,c'] blocks are zero for c != c'
off = 0.0
for c in range(C):
    for c2 in range(C):
        if c != c2:
            off = max(off, A[c*D:(c+1)*D, c2*D:(c2+1)*D].abs().max().item())
print('max off-diagonal block entry :', off)

# per-label blocks built directly
blocks = []
for c in range(C):
    Ac = delta * torch.eye(D) + Z.T @ torch.diag(P[:, c] * (1 - P[:, c])) @ Z
    blocks.append(Ac)
    assert torch.allclose(Ac, A[c*D:(c+1)*D, c*D:(c+1)*D], atol=1e-10)
print('per-label blocks reconstruct A: True')

# eq.(8) full form vs per-label diagonal form
gV = torch.randn(C * D)
z = torch.randn(D)
p = torch.rand(C) * 0.8 + 0.1
s2 = p * (1 - p)
Ps = psi(z)
r = torch.linalg.solve(A, gV)
v = Ps.T @ r
M = Ps.T @ torch.linalg.solve(A, Ps)
full = v @ torch.linalg.solve(torch.diag(1 / s2) + M, v)

# diagonal form: M is diagonal with m_c = z^T A_c^{-1} z
m = torch.tensor([z @ torch.linalg.solve(blocks[c], z) for c in range(C)])
print('M diagonal? off-diag max     :', (M - torch.diag(torch.diag(M))).abs().max().item())
diag_form = (v ** 2 / (1 / s2 + m)).sum()
print('eq.(8) full == diagonal form :', torch.allclose(full, diag_form, atol=1e-9),
      float(full), float(diag_form))

# v_c is normalisation-free: v_c = <R_c, z> with raw augmented z
R = r.view(C, D)
print('v_c == <R_c, z>              :', torch.allclose(v, R @ z, atol=1e-10))
