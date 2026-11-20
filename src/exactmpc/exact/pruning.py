"""Sound pruning of the enumeration by a monotone price bound (statement T6).

STATUS: IMPLEMENTED for a single QP and for a BATCH of QPs sharing (P, G)
(`solve_exact_pruned_batch`, covered by tests/test_pruning.py against full enumeration).
Soundness argument below; the efficiency claim (H6) is empirical and not established.

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


def solve_exact_pruned_batch(
    P: np.ndarray, q: np.ndarray, G: np.ndarray, h: np.ndarray, tol: float = 1e-9, rcond: float = 1e-12
):
    """Batched sound pruning for QPs that share (P, G).

    q: (n,) or (n, B); h: (kappa,) or (kappa, B). The task text wrote the signature as
    `solve_exact_pruned_batch(P, q, h, ...)`: G is a required argument here, since without it
    there is no constrained problem.

    Returns (u, nu, active, visited, total):
      u, nu    (n, B), (kappa, B); nan where no KKT candidate was found (same convention as
               ExactQP.solve); nu is zero outside the active set.
      active   (kappa, B) bool, nu > 0.
      visited  (B,) int, subsets actually evaluated per sample (dependent subsets and subsets
               killed by the bound are not counted; supersets of a dead subset are skipped).
      total    (B,) int, 2**kappa - 1 (the full enumeration size; visited <= total always).

    The per-sample work is independent (dead sets, incumbent and best are per sample), so the
    result is identical to running `solve_exact_pruned` on each column. Everything is shared
    only where it is sample-independent: P^-1, T, r0 and the rank test of each subset.
    """
    P = 0.5 * (np.asarray(P, float) + np.asarray(P, float).T)
    q = np.asarray(q, float)
    h = np.asarray(h, float)
    if q.ndim == 1:
        q = q[:, None]
    if h.ndim == 1:
        h = h[:, None]
    B = max(q.shape[1], h.shape[1])
    q = np.repeat(q, B, axis=1) if q.shape[1] == 1 else q
    h = np.repeat(h, B, axis=1) if h.shape[1] == 1 else h
    Pinv = np.linalg.inv(P)
    k = G.shape[0]
    T = G @ Pinv @ G.T
    ustar = -Pinv @ q
    r0 = G @ ustar - h
    total_value = 2**k - 1
    visited = np.zeros(B, dtype=int)
    best = np.full(B, np.inf)
    nu_best = np.zeros((k, B))
    found = np.zeros(B, bool)
    dead: list[list[int]] = [[] for _ in range(B)]  # bitmasks of dead subsets, per sample

    ok = np.all(r0 <= tol, axis=0)
    best[ok] = 0.0
    found |= ok

    for size in range(1, min(k, P.shape[0]) + 1):
        for J in combinations(range(k), size):
            Jmask = 0
            for j in J:
                Jmask |= 1 << j
            alive = np.array([not any(D & ~Jmask == 0 for D in dead[b]) for b in range(B)], bool)
            # samples whose empty active set is already optimal still count as finished, but the
            # enumeration of the remaining subsets is harmless: best stays 0 and no candidate can
            # beat it, so those samples just accumulate visited counts like the single-QP version.
            if not alive.any():
                continue
            visited[alive] += 1
            Jl = list(J)
            S = T[np.ix_(Jl, Jl)]
            dg = np.sqrt(np.diag(S))
            lam = np.linalg.eigvalsh(S / np.outer(dg, dg))  # correlation matrix: scale invariant
            if lam[0] <= rcond:
                for b in np.where(alive)[0]:
                    dead[b].append(Jmask)  # dependent rows: every superset is dependent too
                continue
            idx = np.where(alive)[0]
            rJ = r0[np.ix_(Jl, idx)]
            nuJ = np.linalg.solve(S, rJ)  # (|J|, |idx|)
            price = 0.5 * np.sum(rJ * nuJ, axis=0)
            viol = r0[:, idx] - T[:, Jl] @ nuJ
            primal_ok = np.all(viol <= tol, axis=0)
            for m, b in enumerate(idx):
                if primal_ok[m]:
                    best[b] = min(best[b], price[m])
                    if np.all(nuJ[:, m] >= -tol) and price[m] < best[b] + tol and price[m] <= best[b]:
                        nu_best[:, b] = 0.0
                        nu_best[Jl, b] = nuJ[:, m]
                        found[b] = True
                if price[m] > best[b] + tol:
                    dead[b].append(Jmask)

    u = ustar - Pinv @ G.T @ nu_best
    u[:, ~found] = np.nan
    nu_best[:, ~found] = np.nan
    return u, nu_best, np.nan_to_num(nu_best) > 0, visited, np.full(B, total_value, dtype=int)
