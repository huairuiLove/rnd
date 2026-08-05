import torch, time
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
from rnd.head import fit, bce
from rnd.scoring import augment

N, C, d = 300, 10, 64
Z = augment(torch.nn.functional.normalize(torch.randn(N, d), dim=-1))
Y = (torch.rand(N, C) > 0.8).double()
t = time.time(); W1 = fit(Z, Y); t1 = time.time() - t
W2 = fit(Z, Y, W0=torch.randn(C, Z.shape[1]) * 0.5)
print('fit %.2fs' % t1)
print('convex: same optimum from different init:', torch.allclose(W1, W2, atol=1e-5),
      float((W1 - W2).abs().max()))
print('loss %.6f vs %.6f' % (bce(Z, Y, W1), bce(Z, Y, W2)))
