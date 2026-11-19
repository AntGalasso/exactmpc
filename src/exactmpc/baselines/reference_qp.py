"""Reference QP solver used in tests (SciPy SLSQP).

STATUS: IMPLEMENTED. Only a correctness reference, not a performance baseline: SLSQP is
accurate to about its own tolerance (around 1e-6 on ill-conditioned problems).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def solve_slsqp(P: np.ndarray, q: np.ndarray, G: np.ndarray, h: np.ndarray, x0: np.ndarray | None = None):
    n = P.shape[0]
    x0 = -np.linalg.solve(P, q) if x0 is None else x0
    cons = [{"type": "ineq", "fun": lambda x: h - G @ x, "jac": lambda x: -G}] if G.shape[0] else []
    res = minimize(lambda x: 0.5 * x @ P @ x + q @ x, x0, jac=lambda x: P @ x + q, constraints=cons, method="SLSQP", options={"ftol": 1e-15, "maxiter": 500})
    return res.x, res.fun, res.success
