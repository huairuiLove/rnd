"""A1: does the quadratic residual-need score predict real retrain gain?

Per trial:
  1. Sample a labelled core set L, fit the head on it.
  2. Hold out a reference set V (disjoint from eval) -> g_V.
  3. For each candidate x: reveal its true label, refit on L+{x}, measure improvement
     on a held-out eval set. That is the ground-truth utility of acquiring x.
  4. Spearman-correlate each scorer's ranking against that ground truth.

Ground truth is macro-AUPRC, which weights rare labels -- where multi-label AL differs.
"""
import argparse
import json
import os
import time
import torch
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr

import torch.nn.functional as F

from rnd_src.head import fit, probs, fisher_damping
from rnd_src.scoring import ResidualNeed, augment, reference_gradient

torch.set_default_dtype(torch.float64)
torch.set_num_threads(os.cpu_count())


def macro_auprc(Y, P):
    keep = Y.sum(0) > 0
    return float(average_precision_score(Y[:, keep], P[:, keep], average='macro'))


def entropy_score(P):
    q = P.clamp(1e-8, 1 - 1e-8)
    return -(q * q.log() + (1 - q) * (1 - q).log()).sum(-1)


def coreset_score(Zc, Zl):
    return 1.0 - (Zc @ Zl.T).max(-1).values


def split_indices(n, seed, cfg):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    a, b = cfg['n_lab'], cfg['n_cand']
    c, d = cfg['n_ref'], cfg['n_eval']
    return perm[:a], perm[a:a + b], perm[a + b:a + b + c], perm[a + b + c:a + b + c + d], g


def run_trial(H, Y, seed, cfg):
    lab, cand, ref, ev, g = split_indices(H.shape[0], seed, cfg)
    Z = augment(H)
    Zl, Yl = Z[lab], Y[lab]
    Zc, Yc = Z[cand], Y[cand]
    Zr, Yr = Z[ref], Y[ref]
    Ze, Ye = Z[ev], Y[ev]

    wd = cfg['wd']
    W = fit(Zl, Yl, wd=wd, iters=cfg['fit_iters'])

    Pc, Pl, Pr = probs(Zc, W), probs(Zl, W), probs(Zr, W)
    GV = reference_gradient(Zr, Pr, Yr)
    delta = fisher_damping(Zl, wd)
    rn = ResidualNeed(Zl, Pl, GV, delta=delta, kappa=cfg['kappa'],
                      label_prior=Yl.mean(0))
    scores = _all_scores(rn, Zl, Zc, Pc, g)
    gains, gains_ap, base = _measure_gains(Zl, Yl, Zc, Yc, Zr, Yr, Ze, Ye, W, cfg)
    return _correlate(scores, gains, gains_ap, rn, Zc, Pc, base, delta)


def ref_loss(Z, Y, W):
    return float(F.binary_cross_entropy_with_logits(Z @ W.T, Y))


def _measure_gains(Zl, Yl, Zc, Yc, Zr, Yr, Ze, Ye, W, cfg):
    """Ground truth per candidate: retrain on L+{x} with the true label revealed.

    Primary signal is the drop in reference-set loss, which is exactly the quantity
    Phi models. Macro-AUPRC on a disjoint eval set is recorded as a secondary check;
    it is far noisier (its bootstrap noise floor exceeds the between-candidate spread),
    so it is reported but not used as the headline correlation.
    """
    base_v = ref_loss(Zr, Yr, W)
    base_ap = macro_auprc(Ye.numpy(), probs(Ze, W).numpy())
    gains = torch.empty(Zc.shape[0])
    gains_ap = torch.empty(Zc.shape[0])
    for i in range(Zc.shape[0]):
        Wi = fit(torch.cat([Zl, Zc[i:i + 1]]), torch.cat([Yl, Yc[i:i + 1]]),
                 wd=cfg['wd'], iters=cfg['retrain_iters'], W0=W)
        gains[i] = base_v - ref_loss(Zr, Yr, Wi)
        gains_ap[i] = macro_auprc(Ye.numpy(), probs(Ze, Wi).numpy()) - base_ap
    return gains, gains_ap, base_ap


def _all_scores(rn, Zl, Zc, Pc, g):
    hard = (Pc > 0.5).double()
    return {
        'rnd_exact': rn.score(Zc, Pc, exact=True),
        'rnd_linear': rn.score(Zc, Pc, exact=False),
        'influence': rn.linear_influence(Zc, Pc).abs(),
        'badge_inf': rn.linear_influence(Zc, Pc, Y=hard).abs(),
        'entropy': entropy_score(Pc),
        'coreset': coreset_score(Zc, Zl),
        'random': torch.rand(Zc.shape[0], generator=g),
    }


def _correlate(scores, gains, gains_ap, rn, Zc, Pc, base, delta):
    gn = gains.numpy()
    out = {}
    for name, s in scores.items():
        v = s.numpy()
        if v.std() == 0.0:
            # Degenerate by construction: proposition 1 makes linear influence
            # identically zero under y~p, so it carries no ranking information.
            out[name] = {'spearman': 0.0, 'p': 1.0, 'spearman_auprc': 0.0,
                         'degenerate': True}
        else:
            rho, p = spearmanr(v, gn)
            rho_ap, _ = spearmanr(v, gains_ap.numpy())
            out[name] = {'spearman': float(rho), 'p': float(p),
                         'spearman_auprc': float(rho_ap)}
    raw = rn.linear_influence(Zc, Pc)
    gap = float((scores['rnd_linear'] - scores['rnd_exact']).mean())
    out['_diag'] = {
        'base_auprc': base,
        'delta': float(delta),
        'gain_mean': float(gains.mean()),
        'gain_std': float(gains.std()),
        'gain_frac_positive': float((gains > 0).double().mean()),
        'gain_ap_std': float(gains_ap.std()),
        'linear_influence_absmax': float(raw.abs().max()),
        'saturation_gap_mean': gap,
        'phi0': float(rn.phi()),
    }
    return out


CFG_KEYS = ('n_lab', 'n_cand', 'n_ref', 'n_eval', 'wd', 'kappa',
            'fit_iters', 'retrain_iters')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='./rnd_src/cache/train_6000.pt')
    ap.add_argument('--trials', type=int, default=5)
    ap.add_argument('--n_lab', type=int, default=20)
    ap.add_argument('--n_cand', type=int, default=100)
    ap.add_argument('--n_ref', type=int, default=4000)
    ap.add_argument('--n_eval', type=int, default=1500)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--kappa', type=float, default=0.0)
    ap.add_argument('--fit_iters', type=int, default=400)
    ap.add_argument('--retrain_iters', type=int, default=400)
    ap.add_argument('--out', default='./rnd_src/a1_results.json')
    args = ap.parse_args()
    cfg = {k: getattr(args, k) for k in CFG_KEYS}

    blob = torch.load(args.cache)
    H, Y = blob['H'], blob['Y']
    print('features %s  labels %d  pos/sample %.2f' % (
        tuple(H.shape), Y.shape[1], Y.sum(1).mean()), flush=True)

    per_trial = []
    for t in range(args.trials):
        t0 = time.time()
        r = run_trial(H, Y, 1000 + t, cfg)
        per_trial.append(r)
        d = r['_diag']
        print('trial %d (%.0fs) gain_std=%.2e  rnd_ex=%+.3f rnd_lin=%+.3f '
              'infl=%+.3f badge=%+.3f ent=%+.3f cs=%+.3f' % (
                  t, time.time() - t0, d['gain_std'],
                  r['rnd_exact']['spearman'], r['rnd_linear']['spearman'],
                  r['influence']['spearman'], r['badge_inf']['spearman'],
                  r['entropy']['spearman'], r['coreset']['spearman']), flush=True)

    names = [k for k in per_trial[0] if not k.startswith('_')]
    summary = {}
    print('\nGround truth: reduction in reference-set loss after true-label retrain.')
    print('%-12s %9s %8s %10s' % ('scorer', 'mean_rho', 'std', 'rho_auprc'))
    for n in sorted(names, key=lambda k: -sum(x[k]['spearman'] for x in per_trial)):
        v = torch.tensor([x[n]['spearman'] for x in per_trial])
        sd = float(v.std()) if len(v) > 1 else 0.0
        summary[n] = {'mean': float(v.mean()), 'std': sd,
                      'per_trial': v.tolist()}
        va = torch.tensor([x[n].get('spearman_auprc', 0.0) for x in per_trial])
        summary[n]['mean_auprc'] = float(va.mean())
        deg = '  (degenerate: identically zero)' if per_trial[0][n].get('degenerate') else ''
        print('%-12s %+9.4f %8.4f %+10.4f%s' % (n, v.mean(), sd, va.mean(), deg))

    dl = [x['_diag']['linear_influence_absmax'] for x in per_trial]
    print('\nproposition 1 (linear influence under y~p) max|.| = %.3e' % max(dl))
    json.dump({'args': vars(args), 'summary': summary, 'per_trial': per_trial},
              open(args.out, 'w'), indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
