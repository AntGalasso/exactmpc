"""Feasible set F of references and the guard set (statement T2, contribution C3).

STATUS: qp_feasible and max_feasible_scale are IMPLEMENTED (tests/test_feasible_set.py).
GuardSet is a STUB: it needs the reference-generic terminal ingredients (Kohler et al. 2019)
or tracking MPC (Limon et al. 2008) to give recursive feasibility under changing references.

F = { theta : exists u with G u <= h(theta) } is convex (projection of a polyhedron), and
A0 is a subset of F. Along a ray from a feasible point the feasible parameters form an
interval [0, t*], so a bisection finds t*.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def qp_feasible(G: np.ndarray, h: np.ndarray) -> bool:
    """True if {u : G u <= h} is non empty (input u is free)."""
    if G.shape[0] == 0:
        return True
    res = linprog(np.zeros(G.shape[1]), A_ub=G, b_ub=h, bounds=[(None, None)] * G.shape[1], method="highs")
    return res.status == 0


def max_feasible_scale(G: np.ndarray, h0: np.ndarray, Hth: np.ndarray, theta0: np.ndarray, direction: np.ndarray, t_max: float = 1e3, iters: int = 50) -> float:
    """Largest t in [0, t_max] with {G u <= h0 + Hth (theta0 + t direction)} non empty."""
    feasible = lambda t: qp_feasible(G, h0 + Hth @ (theta0 + t * direction))
    if not feasible(0.0):
        raise ValueError("theta0 is not feasible")
    if feasible(t_max):
        return t_max
    lo, hi = 0.0, t_max
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if feasible(mid) else (lo, mid)
    return lo


class GuardSet:
    """STUB. A subset R_adm of F, built with terminal ingredients, on which the planner may act."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("T2: guard set with recursive feasibility under reference changes")
