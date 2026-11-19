"""Exact solver by enumeration of active sets (Lin and Liang style) for P positive definite.

STATUS: IMPLEMENTED (covered by tests/test_solver.py: brute force, SLSQP, certificates).

Problem:  min 1/2 u'Pu + q'u   s.t.  G u <= h,     P > 0, G in R^{kappa x n}.

For an active set J (rows with linearly independent G_J):
    u*  = -P^{-1} q,   S_J = G_J P^{-1} G_J',   r_J = G_J u* - h_J
    nu_J = S_J^{-1} r_J,   u_J = u* - P^{-1} G_J' nu_J,   price Delta_J = 1/2 r_J' S_J^{-1} r_J
The optimum is the candidate with u_J primal feasible and nu_J >= 0 (KKT certificate).
Under non-degeneracy (LICQ at the optimum) it is unique. Note: the lowest-price primal feasible
candidate is already the optimum (every feasible point costs at least the optimal price), so
the nu >= 0 test does not change which point is returned; it is kept so that every returned
point is a KKT point and the dual residual in the certificate is meaningful.

Feasibility of a candidate is tested without forming u_J:  G u_J - h = r0 - T[:, J] nu_J,
with T = G P^{-1} G' and r0 = G u* - h, so each subset costs O(kappa |J|) per sample.

`found == False` for a sample means: no KKT candidate exists among the enumerated subsets.
That happens if the QP is infeasible, or if LICQ fails at the optimum (degenerate), or if
max_active is smaller than the number of constraints active at the optimum.

Row scaling: rows of G are normalised to unit norm internally (and the multipliers are mapped
back), so `tol` is a distance to the constraint in the units of u, and the linear independence
test of a subset is invariant to the scale of each row (it is applied to the correlation matrix
of S_J). Without this, rows of very different size (e.g. an input limit and an output limit of a
tiny prediction matrix) are wrongly declared dependent. Reported residuals are in the original
units of G and h.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


@dataclass
class ExactResult:
    u: np.ndarray  # (n, B), nan where not found
    nu: np.ndarray  # (kappa, B), zero outside the active set
    active: np.ndarray  # (kappa, B) bool
    price: np.ndarray  # (B,) Delta = V - V_unc, nan where not found
    found: np.ndarray  # (B,) bool
    primal_violation: np.ndarray  # (B,) max(0, max_j (G u - h)_j)
    dual_violation: np.ndarray  # (B,) max(0, -min nu)
    complementarity: np.ndarray  # (B,) max_j |nu_j (G u - h)_j|
    stationarity: np.ndarray  # (B,) |P u + q + G' nu|_inf
    gap: np.ndarray  # (B,) |primal - dual| / max(1, |primal|)
    subsets_checked: int = 0


class ExactQP:
    """Fixed (P, G); precomputes P^{-1}, T and S_J^{-1} for every independent subset J."""

    def __init__(self, P: np.ndarray, G: np.ndarray, tol: float = 1e-9, max_active: int | None = None, rcond: float = 1e-12):
        P = 0.5 * (np.asarray(P, float) + np.asarray(P, float).T)
        self.P, self.G, self.tol = P, np.asarray(G, float), tol
        self.n, self.kappa = P.shape[0], self.G.shape[0]
        self.d = np.linalg.norm(self.G, axis=1)
        self.d[self.d == 0] = 1.0
        Gn = self.G / self.d[:, None]
        try:
            c = cho_factor(P)
        except np.linalg.LinAlgError as err:
            raise ValueError(
                "P is not positive definite. Use exactmpc.exact.singular (STUB) for P >= 0, "
                "or add a ridge."
            ) from err
        self.Pinv = cho_solve(c, np.eye(self.n))
        self.T = self.G @ self.Pinv @ self.G.T  # original units (used by price.margin_price)
        self.Tn = Gn @ self.Pinv @ Gn.T  # normalised rows (used internally)
        self.max_active = min(self.kappa, self.n) if max_active is None else max_active
        self.subsets: list[tuple[list[int], np.ndarray]] = []
        for size in range(1, self.max_active + 1):
            for J in combinations(range(self.kappa), size):
                J = list(J)
                S = self.Tn[np.ix_(J, J)]
                dg = np.sqrt(np.diag(S))
                lam = np.linalg.eigvalsh(S / np.outer(dg, dg))  # correlation matrix: scale invariant
                if lam[0] > rcond:  # linearly independent rows
                    self.subsets.append((J, np.linalg.inv(S)))

    @property
    def n_candidates(self) -> int:
        """Number of candidates including the empty active set."""
        return 1 + len(self.subsets)

    def solve(self, q: np.ndarray, h: np.ndarray) -> ExactResult:
        """Solve B problems that share (P, G). q: (n,) or (n,B); h: (kappa,) or (kappa,B)."""
        q = np.asarray(q, float)
        h = np.asarray(h, float)
        if q.ndim == 1:
            q = q[:, None]
        if h.ndim == 1:
            h = h[:, None]
        B = max(q.shape[1], h.shape[1])
        q = np.repeat(q, B, axis=1) if q.shape[1] == 1 else q
        h = np.repeat(h, B, axis=1) if h.shape[1] == 1 else h
        tol, k = self.tol, self.kappa

        ustar = -self.Pinv @ q
        r0_orig = self.G @ ustar - h
        r0 = r0_orig / self.d[:, None]  # normalised rows
        best = np.full(B, np.inf)
        nu_best = np.zeros((k, B))
        found = np.zeros(B, bool)

        ok = np.all(r0 <= tol, axis=0)
        best[ok] = 0.0
        found |= ok

        for J, Sinv in self.subsets:
            rJ = r0[J]
            nuJ = Sinv @ rJ
            price = 0.5 * np.sum(rJ * nuJ, axis=0)
            viol = r0 - self.Tn[:, J] @ nuJ  # = (G u_J - h) / d
            valid = (nuJ.min(axis=0) >= -tol) & (viol.max(axis=0) <= tol) & (price < best)
            if valid.any():
                idx = np.where(valid)[0]
                best[idx] = price[idx]
                nu_best[:, idx] = 0.0
                nu_best[np.ix_(J, idx)] = nuJ[:, idx]
                found[idx] = True

        Gn = self.G / self.d[:, None]
        u = ustar - self.Pinv @ Gn.T @ nu_best
        u[:, ~found] = np.nan
        nu_best[:, ~found] = np.nan
        nu_best = nu_best / self.d[:, None]  # multipliers of the ORIGINAL rows
        price = np.where(found, best, np.nan)

        slack = self.G @ u - h
        with np.errstate(invalid="ignore"):
            pv = np.maximum(0.0, np.nanmax(np.where(np.isnan(slack), -np.inf, slack), axis=0)) if k else np.zeros(B)
            dv = np.maximum(0.0, -np.nanmin(np.where(np.isnan(nu_best), np.inf, nu_best), axis=0)) if k else np.zeros(B)
            comp = np.nanmax(np.abs(np.nan_to_num(nu_best) * np.nan_to_num(slack)), axis=0) if k else np.zeros(B)
            stat = np.max(np.abs(self.P @ u + q + self.G.T @ np.nan_to_num(nu_best)), axis=0)
            vp = 0.5 * np.einsum("ib,ij,jb->b", u, self.P, u) + np.sum(q * u, axis=0)
            vd = (
                -0.5 * np.einsum("ib,ij,jb->b", q, self.Pinv, q)
                + np.sum(np.nan_to_num(nu_best) * r0_orig, axis=0)
                - 0.5 * np.einsum("ib,ij,jb->b", np.nan_to_num(nu_best), self.T, np.nan_to_num(nu_best))
            )
            gap = np.abs(vp - vd) / np.maximum(1.0, np.abs(vp))
        for arr in (pv, dv, comp, stat, gap):
            arr[~found] = np.nan
        return ExactResult(
            u=u,
            nu=nu_best,
            active=np.nan_to_num(nu_best) > 0,
            price=price,
            found=found,
            primal_violation=pv,
            dual_violation=dv,
            complementarity=comp,
            stationarity=stat,
            gap=gap,
            subsets_checked=self.n_candidates,
        )
