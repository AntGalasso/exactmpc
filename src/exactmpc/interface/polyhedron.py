"""No-limit polyhedron A0: parameters for which no limit becomes active.

STATUS: IMPLEMENTED (covered by tests/test_interface.py).

The unconstrained input u*(theta) = -P^{-1}(q0 + Qth theta) is affine in theta, so
  A0 = { theta : G u*(theta) <= h0 + Hth theta } = { theta : A theta <= b }
with  A = -(G P^{-1} Qth + Hth),  b = h0 + G P^{-1} q0.
Inside A0 the servo is exactly the unconstrained linear law and the price is zero.
Note: if a planner is confined to A0, the price is identically zero. To use the price the
planner has to leave A0, which requires the feasible set F (see feasible_set.py).
"""

from __future__ import annotations

import numpy as np


def no_limit_polyhedron(Pinv: np.ndarray, G: np.ndarray, q0: np.ndarray, Qth: np.ndarray, h0: np.ndarray, Hth: np.ndarray):
    """Return (A, b) with A0 = {theta: A theta <= b}."""
    A = -(G @ Pinv @ Qth + Hth)
    b = h0 + G @ Pinv @ q0
    return A, b


def in_polyhedron(A: np.ndarray, b: np.ndarray, theta: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Boolean per column of theta."""
    theta = theta if theta.ndim == 2 else theta[:, None]
    return np.all(A @ theta <= b[:, None] + tol, axis=0)
