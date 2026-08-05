"""Residual-need scoring for a single sigmoid classification head.

For a linear head on frozen features, the per-sample BCE gradient is rank one and
the conditional-independence Fisher is block diagonal across labels. Both facts are
checked in _check_blockdiag.py. Consequences used throughout this module:

    A_c = delta*I + sum_x s_c(x)^2 z z^T          (D x D, one block per label)
    v_c(x) = <R_c, z(x)>,  R_c = A_c^{-1} (g_V)_c
    Delta(x)     = sum_c v_c^2 / (1/s_c^2 + m_c)  (exact marginal gain, eq. 8)
    Delta_lin(x) = sum_c s_c^2 v_c^2              (upper bound, eq. 9)

with m_c(x) = z^T A_c^{-1} z.

Block diagonality holds only under the conditional-independence Fisher. Feeding an
empirical label covariance Sigma (section 6) reintroduces cross-label blocks and
invalidates the closed diagonal form -- the two cannot be assumed at the same time.

Two costs drive the implementation:

  * exact scoring needs m_c, i.e. a (D,D) solve per label per candidate, O(N C D^2).
    Eq. (9) needs one (N,D)x(D,C) matmul, O(N C D). Since eq. (9) dominates eq. (8)
    entrywise it is an admissible screen: exact-score a shortlist, and if the best
    discarded upper bound is at or below the best exact value, that pick is provably
    the argmax over the whole set. `select` computes this certificate every step.
  * deflation must not refactorize. A_c changes by a rank-one term, so A_c^{-1} is
    maintained by Sherman-Morrison in O(C D^2) rather than O(C D^3) per pick. At
    C=54, D=769, B=100 that is ~3e9 flops instead of ~2.5e12.

z(x) is h(x) augmented with a constant 1 so the head bias is included. h(x) must be
the exact feature the head consumed: rescaling it (e.g. L2-normalising the CLS after
reading the logits) makes (p-y)z stop being the gradient that produced those logits.
"""
import torch


def augment(H):
    """[h, 1] so the head bias participates in the Fisher geometry."""
    ones = torch.ones(H.shape[0], 1, dtype=H.dtype, device=H.device)
    return torch.cat([H, ones], dim=-1)


def sigma2(P, eps=1e-6):
    """s_c^2 = p_c(1-p_c), floored to keep 1/s^2 finite."""
    return (P * (1.0 - P)).clamp_min(eps)


def fisher_scale(Z, P):
    """Mean diagonal entry of the undamped Fisher block, averaged over labels.

    Lets delta be expressed relative to the geometry it damps. An absolute delta is
    not portable: ||z|| depends on the backbone and on whether features were
    normalised, so the same constant is either negligible or dominant.
    """
    zn2 = (Z ** 2).sum(-1, keepdim=True)
    return float((sigma2(P) * zn2).sum(0).mean() / Z.shape[1])


class ResidualNeed:
    """Maintains per-label blocks A_c and the residual need R (matrix form of A^-1 g_V).

    Z: (N, D) augmented features of the labelled set defining the current posterior.
    P: (N, C) predicted probabilities on those same rows.
    GV: (C, D) reference-set gradient in matrix form.

    After a rare-label floor (eq. 15) the maintained objective is GV_eff^T A^-1 GV_eff
    for a reweighted GV_eff, not the original g_V. See _apply_rare_floor.
    """

    def __init__(self, Z, P, GV, delta=1.0, kappa=0.0, label_prior=None):
        self.C, self.D = GV.shape
        self.delta = float(delta)
        S2 = sigma2(P)
        eye = torch.eye(self.D, dtype=Z.dtype, device=Z.device)
        self.A = torch.einsum('nc,nd,ne->cde', S2, Z, Z) + self.delta * eye
        self.Ainv = torch.cholesky_inverse(torch.linalg.cholesky(self.A))
        self.R = torch.einsum('cij,cj->ci', self.Ainv, GV)
        self.n_floored = 0
        self.floored = False
        if kappa > 0.0 and label_prior is not None:
            self._apply_rare_floor(kappa, label_prior)
        # Effective reference gradient consistent with R after any flooring, so that
        # deflation stays exact: R always equals A^{-1} GV_eff.
        self.GV_eff = torch.einsum('cij,cj->ci', self.A, self.R)

    def _apply_rare_floor(self, kappa, label_prior):
        """Eq. (15): floor the row norm of rare labels so a small V cannot zero them out.

        This changes the objective; it is not a numerical safeguard. Rescaling R_c is
        equivalent to solving the same design problem for a reweighted target
        GV_eff != g_V, so the greedy loop then optimises a rare-label robustified
        objective. Optimality claims hold w.r.t. GV_eff only -- with kappa > 0 the
        result is not the c-optimal design for the original g_V.

        The floor is relative to the median row norm. An absolute kappa is not scale
        invariant: ||R_c|| tracks ||g_V||/delta and spans orders of magnitude across
        datasets and cycles, so a fixed kappa binds on every label or on none.
        kappa=1 means "lift the rarest label to the median".
        """
        w = (label_prior + 1e-8).pow(-0.5)
        w = w / w.max()
        norms = self.R.norm(dim=-1)
        target = kappa * w * norms.median()
        scale = torch.where(norms < target, target / norms.clamp_min(1e-12),
                            torch.ones_like(norms))
        self.R = self.R * scale.unsqueeze(-1)
        self.n_floored = int((scale > 1.0).sum())
        self.floored = self.n_floored > 0

    def v(self, Z):
        """v_c(x) = <R_c, z(x)>. Returns (N, C). O(N C D)."""
        return Z @ self.R.T

    def m(self, Z):
        """m_c(x) = z^T A_c^{-1} z. Returns (N, C).

        O(N C D^2) and allocates (C, N, D): call on a shortlist, never the full pool.
        """
        T = torch.einsum('cde,ne->cnd', self.Ainv, Z)
        return (T * Z.unsqueeze(0)).sum(-1).T.contiguous()

    def score_lin(self, Z, P):
        """Eq. (9). O(N C D) upper bound, used to screen the pool."""
        return (sigma2(P) * self.v(Z) ** 2).sum(-1)

    def score_exact(self, Z, P):
        """Eq. (14'). Exact marginal gain, O(N C D^2) -- shortlists only."""
        V = self.v(Z)
        return (V ** 2 / (1.0 / sigma2(P) + self.m(Z))).sum(-1)

    def score(self, Z, P, exact=True):
        return self.score_exact(Z, P) if exact else self.score_lin(Z, P)

    def linear_influence(self, Z, P, Y=None):
        """Signed linear influence sum_c (p_c - y_c) v_c.

        Y=None takes the expectation under the model's own p, which vanishes by the
        score-function identity (proposition 1). That kills the *signed mean* as a
        ranking signal; it says nothing about |I|, Var(I), or pseudo-label variants.
        """
        V = self.v(Z)
        if Y is None:
            return torch.zeros(Z.shape[0], dtype=Z.dtype, device=Z.device)
        return ((P - Y) * V).sum(-1)

    def deflate(self, z, p):
        """Exact rank-one deflation, eq. (11), per label block.

        A_c += s_c^2 z z^T is rank one, so A_c^{-1} follows by Sherman-Morrison with no
        refactorization. Error accumulates over a batch; `select` logs the
        predicted-vs-realised gap at every step, which is precisely the quantity that
        would drift if it mattered.
        """
        s2 = sigma2(p)
        w = torch.einsum('cij,j->ci', self.Ainv, z)
        mz = (w * z).sum(-1)
        coef = s2 / (1.0 + s2 * mz)
        self.Ainv = self.Ainv - coef.view(-1, 1, 1) * torch.einsum('ci,cj->cij', w, w)
        self.A = self.A + s2.view(-1, 1, 1) * torch.outer(z, z)
        self.R = torch.einsum('cij,cj->ci', self.Ainv, self.GV_eff)

    def phi(self):
        """Phi = GV_eff^T A^-1 GV_eff, summed over label blocks."""
        return (self.GV_eff * self.R).sum()

    def row_norms(self):
        return self.R.norm(dim=-1)


def select(rn, Z, P, budget, exact=True, log=None, topk=32):
    """Greedy eq. (14') argmax with eq. (11) deflation between picks.

    Each step screens the work set with eq. (9) in O(M C D), exact-scores only the top
    `topk`, then checks the certificate: if the best *discarded* upper bound is at or
    below the best exact value, the pick is provably the exact argmax over the work
    set. Steps where the certificate fails are counted, not hidden -- that count is the
    honest statement of how approximate the greedy path was.

    Returns (picks, trace). trace[b] holds the predicted gain for the chosen sample and
    the realised drop in Phi. Those must agree to solver precision; the gap is the
    cheapest check that R stays equal to A^{-1} GV_eff through Sherman-Morrison, so it
    is logged rather than asserted.
    """
    picks, trace = [], []
    n = Z.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=Z.device)
    for b in range(budget):
        ub = rn.score_lin(Z, P)
        ub[~alive] = -float('inf')
        if exact:
            k = min(topk, int(alive.sum()))
            cand = ub.topk(k).indices
            ex = rn.score_exact(Z[cand], P[cand])
            j = int(ex.argmax())
            i = int(cand[j])
            predicted = float(ex[j])
            rest = ub.clone()
            rest[cand] = -float('inf')
            best_dropped = float(rest.max())
            certified = bool(best_dropped <= predicted)
        else:
            i = int(ub.argmax())
            predicted = float(ub[i])
            best_dropped, certified = float('nan'), None
        phi_before = float(rn.phi())
        rn.deflate(Z[i], P[i])
        phi_after = float(rn.phi())
        alive[i] = False
        picks.append(i)
        rec = {'step': b, 'idx': i, 'predicted': predicted,
               'phi_before': phi_before, 'phi_after': phi_after,
               'phi_drop': phi_before - phi_after, 'certified': certified,
               'best_dropped_ub': best_dropped,
               'rel_err': abs((phi_before - phi_after) - predicted) / max(predicted, 1e-300)}
        trace.append(rec)
        if log is not None and (b < 3 or b == budget - 1):
            log('  [rnd] step %d idx=%d pred=%.6e phi_drop=%.6e rel_err=%.2e certified=%s'
                % (b, i, predicted, rec['phi_drop'], rec['rel_err'], certified))
    return picks, trace


def reference_gradient(Z, P, Y):
    """g_V in matrix form (C, D): mean BCE gradient over the reference set."""
    return (P - Y).T @ Z / Z.shape[0]
