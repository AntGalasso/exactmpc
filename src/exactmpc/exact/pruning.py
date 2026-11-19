"""Sound pruning of the enumeration by a monotone price bound (statement T6).

STATUS: IMPLEMENTED for a single QP (covered by tests/test_pruning.py against full
enumeration). Soundness argument below; the efficiency claim (H6) is empirical and not
established. Not batched.

Argument. For independent rows, the equality-constrained price Delta_J = 1/2 r_J' S_J^{-1} r_J
is the minimum of V - V_unc over {G_J u = h_J}. Adding equality constraints cannot lower a
minimum, so D subset of J implies Delta_J >= Delta_D. The optimal active set J* has
Delta_{J*} = Delta* (the optimal price), and every primal feasible candidate has price
>= Delta*. Hence once some primal feasible candidate with price `incumbent` is known, any D with
Delta_D > incumbent >= Delta* cannot be J*, and neither can any superset of D.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def solve_exact_pruned(P: np.ndarray, q: np.ndarray, G: np.ndarray, h: np.ndarray, tol: float = 1e-9, rcond: float = 1e-12):
    """Return (u, nu, active_set, visited, total) for one QP. visited counts subsets evaluated."""
    P = 0.5 * (P + P.T)
    Pinv = np.linalg.inv(P)
    k = G.shape[0]
    T = G @ Pinv @ G.T
    ustar = -Pinv @ q
    r0 = G @ ustar - h
    total = 2**k - 1
    visited = 0
    if np.all(r0 <= tol):
        return ustar, np.zeros(k), (), visited, total
    incumbent = np.inf
    dead: list[set[int]] = []
    best = None
    for size in range(1, min(k, P.shape[0]) + 1):
        for J in combinations(range(k), size):
            Jset = set(J)
            if any(D <= Jset for D in dead):
                continue
            Jl = list(J)
            S = T[np.ix_(Jl, Jl)]
            dg = np.sqrt(np.diag(S))
            lam = np.linalg.eigvalsh(S / np.outer(dg, dg))  # correlation matrix: scale invariant
            visited += 1
            if lam[0] <= rcond:
                dead.append(Jset)  # dependent rows: every superset is dependent too
                continue
            rJ = r0[Jl]
            nu = np.linalg.solve(S, rJ)
            price = 0.5 * rJ @ nu
            primal_ok = np.all(r0 - T[:, Jl] @ nu <= tol)
            if primal_ok:
                incumbent = min(incumbent, price)
                if np.all(nu >= -tol):
                    best = (Jl, nu, price)
            if price > incumbent + tol:
                dead.append(Jset)
    if best is None:
        return None, None, None, visited, total
    Jl, nu, _ = best
    nu_full = np.zeros(k)
    nu_full[Jl] = nu
    return ustar - Pinv @ G.T @ nu_full, nu_full, tuple(Jl), visited, total
