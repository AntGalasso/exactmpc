"""Singular Hessian (P >= 0): over-actuated axes, zero input weight (contribution C4).

STATUS: IMPLEMENTED (gate G2). Exact solver by enumeration of active sets for P >= 0,
with spectral decomposition, minimum-norm selection on the optimal face, and KKT
certificate. Covered by tests/test_singular.py.

Problem:  min 1/2 u'Pu + q'u   s.t.  G u <= h,   P = P' >= 0, G in R^{kappa x n}.

Spectral decomposition P = V Lambda V' with V (n x r) orthonormal basis of range(P),
Lambda > 0, N (n x (n-r)) orthonormal basis of null(P), P^+ = V Lambda^{-1} V'.

For each active set J the KKT system M_J [u; nu_J] = [-q; h_J],
M_J = [[P, G_J'], [G_J, 0]] is solved in least squares. Consistency, primal and dual
feasibility are required. Minimum-norm u is selected on the optimal face.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .solver import ExactResult
from ..interface.feasible_set import qp_feasible


def _null_space(M: np.ndarray, rcond: float) -> np.ndarray:
    """Orthonormal basis (columns) of the null space of M."""
    if M.size == 0:
        return np.eye(M.shape[1] if M.ndim == 2 else 0)
    u, s, vt = np.linalg.svd(M, full_matrices=True)
    if s.size == 0 or s[0] == 0:
        return vt.T.copy()
    threshold = rcond * s[0]
    rank = int(np.sum(s > threshold))
    return vt[rank:].T.copy()


class ExactSingularQP:
    """Fixed (P, G) with P >= 0; precomputes spectral factors and candidate sets."""

    def __init__(self, P, G, tol: float = 1e-9, rcond: float = 1e-12, max_active=None):
        P = 0.5 * (np.asarray(P, float) + np.asarray(P, float).T)
        self.P = P
        self.G = np.asarray(G, float)
        self.tol = float(tol)
        self.rcond = float(rcond)
        self.n = P.shape[0]
        self.kappa = self.G.shape[0]
        if max_active is None:
            max_active = min(self.kappa, self.n)
        self.max_active = int(max_active)

        # spectral decomposition P = V Lambda V'
        w, V = np.linalg.eigh(P)
        # sort descending for convenience (not strictly required)
        order = np.argsort(w)[::-1]
        w = w[order]
        V = V[:, order]
        if w.size == 0 or np.max(w) <= 0:
            pos = np.zeros(w.shape, bool)
        else:
            pos = w > (rcond * float(np.max(w)))
        # guard: if all zero, pos all False
        if w.size == 0:
            pos = np.zeros(0, bool)
        self.V = V[:, pos]  # n x r
        self.Lambda = w[pos]
        self.r = self.V.shape[1]
        null_mask = ~pos
        self.N = V[:, null_mask]  # n x (n-r)
        if self.r > 0:
            self.Pplus = self.V @ np.diag(1.0 / self.Lambda) @ self.V.T
        else:
            self.Pplus = np.zeros((self.n, self.n))

        # candidate subsets J (as sorted tuples), sizes 0..max_active
        self._subsets: list[tuple[int, ...]] = []
        for size in range(0, self.max_active + 1):
            for J in combinations(range(self.kappa), size):
                self._subsets.append(tuple(J))

        # precompute null spaces of M_J (sample independent)
        self._null_cache: dict[tuple[int, ...], np.ndarray] = {}
        for J in self._subsets:
            self._null_cache[J] = self._null_of(J)

    @property
    def n_candidates(self) -> int:
        return len(self._subsets)

    def _null_of(self, J):
        s = len(J)
        GJ = self.G[list(J), :] if s > 0 else np.zeros((0, self.n))
        # M_J = [[P, GJ'], [GJ, 0]]
        M = np.block([[self.P, GJ.T],
                      [GJ, np.zeros((s, s))]]) if s > 0 else self.P.copy()
        if s == 0:
            M = self.P.copy()
        return _null_space(M, self.rcond)

    def _solve_subset(self, J, q, h):
        """Return per-sample u (n,B), nu (s,B) in original units for consistent samples; validity mask.

        q: (n,B), h: (kappa,B).
        """
        B = q.shape[1]
        s = len(J)
        GJ = self.G[list(J), :] if s > 0 else np.zeros((0, self.n))
        # M_J
        if s > 0:
            M = np.block([[self.P, GJ.T],
                          [GJ, np.zeros((s, s))]])
        else:
            M = self.P.copy()
        # RHS b = [-q; h_J]
        if s > 0:
            Bmat = np.vstack([-q, h[list(J), :]])  # (n+s, B)
        else:
            Bmat = -q.copy()  # (n, B)
        # lstsq
        X0, *_ = np.linalg.lstsq(M, Bmat, rcond=self.rcond)
        # one step of iterative refinement for accuracy on ill-conditioned KKT systems
        R = M @ X0 - Bmat
        dX, *_ = np.linalg.lstsq(M, R, rcond=self.rcond)
        X0 = X0 - dX
        R = M @ X0 - Bmat
        resnorm = np.linalg.norm(R, axis=0)
        bnorm = np.linalg.norm(Bmat, axis=0)
        consistent = resnorm / (1.0 + bnorm) <= 1e-8

        Z = self._null_cache[J]  # (n+s, z)
        # per-sample min-norm selection
        u_out = np.full((self.n, B), np.nan)
        nu_out = np.zeros((s, B))  # for J entries; caller maps to full
        # we return nu for this subset's rows only
        valid = np.zeros(B, bool)
        # precompute Z_u = Z[:n, :]
        Zu = Z[:self.n, :] if Z.shape[0] >= self.n else Z
        for b in range(B):
            if not consistent[b]:
                continue
            x0 = X0[:, b].copy()
            if Z.shape[1] > 0 and Zu.shape[1] > 0:
                t, *_ = np.linalg.lstsq(Zu, -x0[:self.n], rcond=self.rcond)
                x = x0 + Z @ t
            else:
                x = x0
            u_b = x[:self.n]
            nu_b = x[self.n:] if s > 0 else np.zeros(0)
            u_out[:, b] = u_b
            if s > 0:
                nu_out[:, b] = nu_b
            valid[b] = True
        return u_out, nu_out, valid, consistent

    def solve(self, q, h) -> ExactResult:
        q = np.asarray(q, float)
        h = np.asarray(h, float)
        if q.ndim == 1:
            q = q[:, None]
        if h.ndim == 1:
            h = h[:, None]
        B = max(q.shape[1], h.shape[1])
        q = np.repeat(q, B, axis=1) if q.shape[1] == 1 else q
        h = np.repeat(h, B, axis=1) if h.shape[1] == 1 else h
        tol = self.tol
        k = self.kappa

        # best tracking
        best_price = np.full(B, np.inf)
        best_norm = np.full(B, np.inf)
        best_u = np.full((self.n, B), np.nan)
        best_nu_full = np.zeros((k, B))  # original rows, filled for found
        best_J = [None] * B
        found = np.zeros(B, bool)

        V_unc = -0.5 * np.einsum("ib,ij,jb->b", q, self.Pplus, q)

        # enumerate candidates in order: size asc, then lex
        for J in self._subsets:
            u_cand, nu_cand_sub, valid_mask, consistent_mask = self._solve_subset(J, q, h)
            # map nu subset to full
            # compute primal feasibility per sample
            # slack = G @ u - h
            # for invalid (nan) handle
            # compute price for all samples
            price_arr = np.full(B, np.nan)
            # stationarity etc not needed for validity? compute pv, dv
            pv_arr = np.full(B, np.nan)
            dv_arr = np.full(B, np.nan)
            stat_arr = np.full(B, np.nan)
            for b in range(B):
                if not (valid_mask[b] and consistent_mask[b]):
                    continue
                ub = u_cand[:, b]
                if np.any(np.isnan(ub)):
                    continue
                # Project onto range(P) when feasible, to remove null(P) components
                # (minimum-norm on the optimal face) without changing P u.
                if self.r > 0:
                    ub_proj = self.V @ (self.V.T @ ub)
                    slack_proj = self.G @ ub_proj - h[:, b]
                    pv_proj = max(0.0, float(np.max(slack_proj))) if k > 0 else 0.0
                    delta = ub - ub_proj
                    GJ = self.G[list(J), :] if len(J) > 0 else np.zeros((0, self.n))
                    free_delta = (GJ.shape[0] == 0) or (np.linalg.norm(GJ @ delta) <= 1e-6 * (1.0 + np.linalg.norm(delta)))
                    if pv_proj <= tol + 1e-7 and free_delta:
                        ub = ub_proj
                        u_cand[:, b] = ub
                # primal violation
                slack_b = self.G @ ub - h[:, b]
                pv = float(np.max(slack_b)) if k > 0 else 0.0
                pv = max(0.0, pv)
                pv_arr[b] = pv
                # nu full for this candidate
                nu_full_b = np.zeros(k)
                if len(J) > 0:
                    nu_full_b[list(J)] = nu_cand_sub[:, b]
                # remap? already original since M uses original G
                dv = float(np.max(np.concatenate(([0.0], -nu_full_b)))) if k > 0 else 0.0
                # min nu
                if k > 0:
                    dv = max(0.0, -float(np.min(nu_full_b)))
                else:
                    dv = 0.0
                dv_arr[b] = dv
                # stationarity
                stat = float(np.max(np.abs(self.P @ ub + q[:, b] + self.G.T @ nu_full_b))) if self.n > 0 else 0.0
                stat_arr[b] = stat
                # price
                Vb = 0.5 * (ub @ self.P @ ub) + q[:, b] @ ub
                price_arr[b] = Vb - V_unc[b]

            # validity: consistent & primal feasible & dual feasible & stationarity
            for b in range(B):
                if not (valid_mask[b] and consistent_mask[b]):
                    continue
                if np.isnan(price_arr[b]):
                    continue
                if pv_arr[b] > tol + 1e-15:
                    continue
                if dv_arr[b] > tol + 1e-15:
                    continue
                if stat_arr[b] > 1e-8 + 1e-12:
                    continue
                # candidate valid
                better = False
                if not found[b]:
                    better = True
                else:
                    if price_arr[b] < best_price[b] - 1e-12:
                        better = True
                    elif abs(price_arr[b] - best_price[b]) <= 1e-9:
                        nrm = float(np.linalg.norm(u_cand[:, b]))
                        if nrm < best_norm[b] - 1e-12:
                            better = True
                        elif abs(nrm - best_norm[b]) <= 1e-12:
                            if best_J[b] is None or J < best_J[b]:
                                better = True
                if better:
                    found[b] = True
                    best_price[b] = price_arr[b]
                    best_norm[b] = float(np.linalg.norm(u_cand[:, b]))
                    best_u[:, b] = u_cand[:, b]
                    nu_full_b = np.zeros(k)
                    if len(J) > 0:
                        nu_full_b[list(J)] = nu_cand_sub[:, b]
                    best_nu_full[:, b] = nu_full_b
                    best_J[b] = J

        # Build u, nu with nan for not found
        u = best_u.copy()
        nu = best_nu_full.copy()
        u[:, ~found] = np.nan
        nu[:, ~found] = np.nan

        price = np.where(found, best_price, np.nan)

        # certificates
        slack = self.G @ np.nan_to_num(u) - h  # but for nan u, nan_to_num 0 -> garbage; handle
        # compute slack properly with nan propagation
        with np.errstate(invalid="ignore"):
            slack = self.G @ u - h
        pv = np.full(B, np.nan)
        dv = np.full(B, np.nan)
        comp = np.full(B, np.nan)
        stat = np.full(B, np.nan)
        gap = np.full(B, np.nan)
        for b in range(B):
            if not found[b]:
                continue
            ub = u[:, b]
            nub = nu[:, b]
            sl = self.G @ ub - h[:, b]
            pv[b] = max(0.0, float(np.max(sl))) if k > 0 else 0.0
            dv[b] = max(0.0, float(-np.min(nub))) if k > 0 else 0.0
            comp[b] = float(np.max(np.abs(nub * sl))) if k > 0 else 0.0
            stat[b] = float(np.max(np.abs(self.P @ ub + q[:, b] + self.G.T @ nub)))
            Vb = 0.5 * (ub @ self.P @ ub) + q[:, b] @ ub
            # gap
            w = self.N.T @ (q[:, b] + self.G.T @ nub)
            if np.linalg.norm(w) > 1e-8:
                gap[b] = np.nan
            else:
                gnu = q[:, b] + self.G.T @ nub
                # (q+G'nu) = -P u at a KKT point, so (q+G'nu)' P+ (q+G'nu) = u' P u
                A = float(ub @ self.P @ ub)
                dual_val = -h[:, b] @ nub - 0.5 * A
                gap[b] = abs(Vb - dual_val) / max(1.0, abs(Vb))

        active = np.nan_to_num(nu) > 0

        # infeasibility: if not found but polyhedron nonempty, still found=False (convention).
        # (No change needed.)

        return ExactResult(
            u=u,
            nu=nu,
            active=active,
            price=price,
            found=found,
            primal_violation=pv,
            dual_violation=dv,
            complementarity=comp,
            stationarity=stat,
            gap=gap,
            subsets_checked=self.n_candidates,
        )


def solve_exact_singular(P, q, G, h, tol: float = 1e-9, rcond: float = 1e-12, max_active=None) -> ExactResult:
    qp = ExactSingularQP(P, G, tol=tol, rcond=rcond, max_active=max_active)
    return qp.solve(q, h)
