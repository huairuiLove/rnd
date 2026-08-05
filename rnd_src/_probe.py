import os, torch
torch.set_default_dtype(torch.float64)
torch.set_num_threads(os.cpu_count())
import torch.nn.functional as F
from rnd_src.head import fit, probs, fisher_damping
from rnd_src.scoring import augment, ResidualNeed, reference_gradient
from rnd_src.a1 import split_indices
from scipy.stats import spearmanr

b=torch.load('rnd_src/cache/train_6000.pt'); H,Y=b['H'],b['Y']
Z=augment(H)

def trial(seed, n_lab=20, n_cand=100, n_ref=4000, wd=1e-4):
    cfg=dict(n_lab=n_lab,n_cand=n_cand,n_ref=n_ref,n_eval=1,wd=wd,kappa=0,fit_iters=400,retrain_iters=400)
    lab,cand,ref,ev,g=split_indices(H.shape[0],seed,cfg)
    Zl,Yl,Zc,Yc,Zr,Yr=Z[lab],Y[lab],Z[cand],Y[cand],Z[ref],Y[ref]
    W=fit(Zl,Yl,wd=wd,iters=400)
    def L_V(Wm): return float(F.binary_cross_entropy_with_logits(Zr@Wm.T,Yr))
    base=L_V(W)
    Pc,Pl,Pr=probs(Zc,W),probs(Zl,W),probs(Zr,W)
    GV=reference_gradient(Zr,Pr,Yr)
    delta=fisher_damping(Zl,wd)
    rn=ResidualNeed(Zl,Pl,GV,delta=delta)
    exact=rn.score(Zc,Pc,exact=True); lin=rn.score(Zc,Pc,exact=False)
    d=torch.empty(n_cand)
    for i in range(n_cand):
        Wi=fit(torch.cat([Zl,Zc[i:i+1]]),torch.cat([Yl,Yc[i:i+1]]),wd=wd,iters=400,W0=W)
        d[i]=base-L_V(Wi)
    r_ex,_=spearmanr(exact.numpy(),d.numpy())
    r_lin,_=spearmanr(lin.numpy(),d.numpy())
    entropy = -(Pc.clamp(1e-8,1-1e-8).log()*Pc + (1-Pc).clamp(1e-8,1-1e-8).log()*(1-Pc)).sum(-1)
    r_ent,_=spearmanr(entropy.numpy(),d.numpy())
    r_rand,_=spearmanr(torch.rand(n_cand,generator=g).numpy(), d.numpy())
    print('seed=%d  gain_std=%.2e  rnd_exact=%+.3f rnd_lin=%+.3f entropy=%+.3f random=%+.3f'%(seed,d.std(),r_ex,r_lin,r_ent,r_rand))
    return r_ex,r_lin,r_ent,r_rand

res=[trial(1000+t) for t in range(8)]
import numpy as np
res=np.array(res)
print('\nmeans: rnd_exact=%.3f rnd_lin=%.3f entropy=%.3f random=%.3f'%tuple(res.mean(0)))
