"""RND acquisition against a live CoMAL model.

Maps section 12 of the design doc onto CoMAL's AL loop. Unlike rnd_src/a1.py, which fits
its own proxy head, this reads the trained backbone's own features and head: the Fisher
geometry has to describe the posterior the AL loop actually holds, not a surrogate.

Coordinate system: features are the raw CLS vectors, NOT L2-normalised. The whole
derivation treats (p - y) z as the head's own gradient, which is only true in the
coordinates the head consumed. Normalising CLS after reading the logits silently breaks
that -- p comes from clf(cls) while the gradient is formed at clf(cls/||cls||), so the
Fisher describes a head that was never trained. `--rnd_normalize` restores the old
behaviour by refitting the head on normalised features, which is the other consistent
option; it is an ablation, not the default.

The reference set V is CoMAL's validation split. It supplies only the design direction
g_V; the test split stays closed throughout.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from samplers import SubsetSequentialSampler
from rnd_src.scoring import (ResidualNeed, augment, fisher_scale,
                             reference_gradient, select)


@torch.no_grad()
def embed(model, dataset, indices, device, batch_size=32, num_workers=0):
    """Current-encoder CLS features, logits and ground-truth labels for `indices`.

    Features are returned in the head's own coordinates (raw CLS, un-normalised) so
    that logits == clf(cls) holds exactly and (p - y) z is the true head gradient.
    Order follows `indices` exactly (sequential sampler), so row i corresponds to
    indices[i]. Returns float64 to match the scoring module.
    """
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True, sampler=SubsetSequentialSampler(indices))
    H, L, Y = [], [], []
    for text_inputs, labels, _, _, _ in loader:
        ids, type_ids, mask = [x.to(device) for x in text_inputs]
        logits, _, cls = model([ids, type_ids, mask, None])
        H.append(cls.double().cpu())
        L.append(logits.double().cpu())
        Y.append(labels.double())
    return torch.cat(H), torch.cat(L), torch.cat(Y)


def refit_head(Zl, Yl, wd, iters=200):
    """Refit the linear head on the given coordinates, returning fresh logits fn.

    Only used by --rnd_normalize: if features are rescaled, the head must be refitted
    in the new coordinates or the gradient identity breaks. Returns W (C, D) for
    augmented Z, so logits are Z @ W.T.
    """
    from rnd_src.head import fit
    return fit(Zl, Yl, wd=wd, iters=iters)


def _log_geometry(rn, Zl, Pl, GV, delta, label_prior, log):
    """Print the quantities that make the derivation falsifiable at runtime.

    Every line here corresponds to a specific claim in the design doc, so a broken
    port shows up in the log rather than as a silently mediocre AL curve.
    """
    log('[rnd] geometry: C=%d d=%d |L|=%d delta=%.4e' % (rn.C, rn.D, Zl.shape[0], delta))
    log('[rnd] Phi_0 = %.6e   ||g_V||_F = %.4e' % (float(rn.phi()), float(GV.norm())))
    rows = rn.row_norms()
    log('[rnd] ||R_c||: min=%.3e med=%.3e max=%.3e' % (
        float(rows.min()), float(rows.median()), float(rows.max())))
    # Proposition 1: the linear influence under y~p must vanish identically.
    zero = rn.linear_influence(Zl, Pl).abs().max()
    log('[rnd] prop.1 check  max|linear influence under y~p| = %.3e' % float(zero))
    # Rarest labels are where eq. (15) either helps or is absent; show both.
    order = label_prior.argsort()
    rare = order[:5].tolist()
    log('[rnd] rarest labels %s  prior=%s  ||R_c||=%s' % (
        rare,
        ['%.2e' % float(label_prior[c]) for c in rare],
        ['%.2e' % float(rows[c]) for c in rare]))
    # A degenerate prior makes eq. (15) a no-op regardless of kappa; say so loudly
    # rather than letting the ablation report "no effect" as a finding.
    spread = float(label_prior.max() / label_prior.min().clamp_min(1e-12))
    log('[rnd] label prior spread max/min = %.1f' % spread)
    if spread < 1.5:
        log('[rnd] WARNING label prior nearly flat -- eq.(15) floor cannot '
            'differentiate labels; enlarge --rnd_ref_size')
    log('[rnd] eq.(15) floor: %d/%d label rows lifted' % (rn.n_floored, rn.C))
    if rn.floored:
        # Flooring reweights the target, so Phi is no longer the c-optimal objective for
        # g_V. Say so in the log, otherwise the Phi trace reads as an exact c-optimal run.
        log('[rnd] objective: rare-label robustified (GV_eff != g_V, '
            '||GV_eff-g_V||/||g_V|| = %.3e); Phi is optimal w.r.t. GV_eff, not g_V'
            % float((rn.GV_eff - GV).norm() / GV.norm().clamp_min(1e-300)))
    else:
        log('[rnd] objective: c-optimal for g_V (no flooring applied)')


def _log_selection(trace, scores, picks, n_pool, n_work, budget, exact, log):
    """Report deflation health and how much of the pool was actually considered."""
    rel = max(t['rel_err'] for t in trace)
    if exact:
        log('[rnd] deflation exactness: max rel err over %d steps = %.3e' % (len(trace), rel))
        if rel > 1e-6:
            log('[rnd] WARNING deflation not exact -- R != A^{-1} GV_eff. Sherman-Morrison '
                'accumulates error over the batch; raise --rnd_delta or shrink the budget')
        drift = trace[0].get('m_drift')
        if drift is not None:
            # Cached m is carried by rank-one updates instead of being recomputed each
            # step; this is the measurement of how far that drifted over the batch.
            log('[rnd] cached-m drift: max rel err = %.3e' % drift)
            if drift > 1e-6:
                log('[rnd] WARNING cached m has drifted -- exact scores are stale; '
                    'shrink the budget or raise damping')
    else:
        # Eq. (9) is an upper bound on the realised drop, so predicted > phi_drop is
        # expected here; the gap size is the saturation effect of corollary 2.1.
        log('[rnd] eq.(9) ablation: mean predicted/actual ratio = %.3f' % (
            sum(t['predicted'] / max(t['phi_drop'], 1e-300) for t in trace) / len(trace)))
    phis = [trace[0]['phi_before']] + [t['phi_after'] for t in trace]
    mono = all(b >= a - 1e-12 for a, b in zip(phis[1:], phis[:-1]))
    log('[rnd] Phi: %.6e -> %.6e (total drop %.3f%%), monotone=%s' % (
        phis[0], phis[-1], 100.0 * (phis[0] - phis[-1]) / max(phis[0], 1e-300), mono))
    # A silent top-M truncation would make coverage look like full-pool coverage.
    log('[rnd] pool=%d  deflation work set=%d  picked=%d  (dropped %d candidates '
        'before deflation)' % (n_pool, n_work, len(picks), n_pool - n_work))
    s = scores[scores > -float('inf')]
    log('[rnd] score dist: min=%.3e med=%.3e max=%.3e' % (
        float(s.min()), float(s.median()), float(s.max())))


def acquire(args, models, dataset, val_dataset, pool, labeled, device, log=print):
    """Select args.sample_pair_num samples from `pool` by eq. (14') + eq. (11).

    pool / labeled are index lists into `dataset`. Returns positions into `pool`.
    """
    model = models['backbone']
    model.eval()
    bs, nw = args.batch_size, 0

    n_ref = min(args.rnd_ref_size, len(val_dataset))
    ref = list(range(n_ref))
    Hl, Ll, Yl = embed(model, dataset, list(labeled), device, bs, nw)
    Hr, Lr, Yr = embed(model, val_dataset, ref, device, bs, nw)
    Hu, Lu, _ = embed(model, dataset, list(pool), device, bs, nw)

    Zl, Zr, Zu = augment(Hl), augment(Hr), augment(Hu)
    if args.rnd_normalize:
        # Ablation branch. Rescaling features invalidates the live head, so the head is
        # refitted in the new coordinates and probabilities recomputed from it --
        # otherwise p and the gradient would live in different spaces.
        Zl, Zr, Zu = (F.normalize(Z, dim=-1) for Z in (Zl, Zr, Zu))
        W = refit_head(Zl, Yl, args.rnd_wd)
        Pl, Pr, Pu = (torch.sigmoid(Z @ W.T) for Z in (Zl, Zr, Zu))
        log('[rnd] coords: L2-normalised features, head refitted (ablation)')
    else:
        Pl, Pr, Pu = torch.sigmoid(Ll), torch.sigmoid(Lr), torch.sigmoid(Lu)
        # Cheapest possible check that features and probabilities share a coordinate
        # system: the head applied to Z must reproduce the logits it emitted.
        Wl = torch.cat([model.clf.weight.double().cpu(),
                        model.clf.bias.double().cpu().unsqueeze(-1)], dim=-1)
        mism = float((Zl @ Wl.T - Ll).abs().max())
        log('[rnd] coords: raw CLS, max|Z W^T - logits| = %.3e' % mism)
        if mism > 1e-6:
            log('[rnd] WARNING features and logits disagree -- (p-y)z is not the '
                'head gradient; check for feature rescaling in embed()')

    GV = reference_gradient(Zr, Pr, Yr)
    # delta is a regularisation choice, not a derived quantity. CoMAL trains its head
    # with AdamW, which has no L2 term to match, so the 2*wd*|L|/d form is only a
    # convention inherited from the proxy head in rnd_src/head.py. What it actually does is
    # set the scale of the null space: with |L| << d the blocks are rank-deficient and
    # delta decides how much a direction unseen in L still counts. Expressed relative
    # to the Fisher's own scale so the same number transfers across backbones.
    fs = fisher_scale(Zl, Pl)
    if args.rnd_delta > 0.0:
        delta = args.rnd_delta
    elif args.rnd_delta_rel > 0.0:
        delta = args.rnd_delta_rel * fs
    else:
        delta = 2.0 * args.rnd_wd * Zl.shape[0] / Zl.shape[1]
    log('[rnd] delta=%.4e (fisher scale=%.4e, ratio=%.3e) -- tuning knob, not derived'
        % (delta, fs, delta / max(fs, 1e-300)))
    # Prior comes from V, not L: with |L| ~ 100 and C = 54 most labels have zero
    # positives in L, which would flatten eq. (15)'s weights to a single value and
    # silently turn the rare-label floor into a no-op. V is design-side, so this is
    # allowed. Laplace smoothing keeps w_c finite for labels absent from V too.
    label_prior = (Yr.sum(0) + 1.0) / (Yr.shape[0] + 2.0)
    rn = ResidualNeed(Zl, Pl, GV, delta=delta, kappa=args.rnd_kappa,
                      label_prior=label_prior)
    _log_geometry(rn, Zl, Pl, GV, delta, label_prior, log)

    budget = min(args.sample_pair_num, Zu.shape[0])
    exact = not args.rnd_linear_score
    # Never exact-score the pool: eq.(8) is O(|U| C d^2) and allocates (C,|U|,d), which
    # is what made the previous version unaffordable. score_ub is O(|U| C d) and upper
    # bounds eq.(8) entrywise, so a top-M cut on it can only drop candidates whose exact
    # score was also below the cut -- the truncation is admissible, not merely cheap.
    scores = rn.score_ub(Zu, Pu)
    work = min(max(args.rnd_work_mult * budget, budget), Zu.shape[0])
    top = scores.topk(work).indices
    if exact and work < Zu.shape[0]:
        # Certificate for the screen: compare the best discarded upper bound against
        # the best kept exact value. If it does not hold, the first pick may not be the
        # pool argmax and the work set is too small -- reported either way.
        rest = scores.clone()
        rest[top] = -float('inf')
        best_cut, best_kept = float(rest.max()), float(rn.score_exact(Zu[top], Pu[top]).max())
        log('[rnd] screen certificate: best cut ub=%.4e vs best kept exact=%.4e -> '
            'first pick provably pool-exact: %s'
            % (best_cut, best_kept, best_cut <= best_kept))
    picks_local, trace = select(rn, Zu[top], Pu[top], budget, exact=exact, log=log)
    picks = [int(top[i]) for i in picks_local]
    _log_selection(trace, scores, picks, Zu.shape[0], work, budget, exact, log)
    return picks, trace
