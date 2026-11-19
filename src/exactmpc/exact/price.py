"""Price, its gradient, and the price of a constraint margin.

STATUS: IMPLEMENTED (covered by tests/test_price.py: finite differences, continuity across
active-set changes (T1, numerical evidence only), and margin price against a re-solve (T4)).

Parameter theta enters the QP affinely:  q = q0 + Qth theta,  h = h0 + Hth theta.

  V(theta)      = min 1/2 u'Pu + q'u        (QP part of the cost, constant s excluded)
  Delta(theta)  = V - V_unc,  V_unc = -1/2 q'P^{-1}q = 1/2 r_J' S_J^{-1} r_J on the optimum
  dV/dtheta     = Qth' u  - Hth' nu                      (envelope theorem)
  dDelta/dtheta = -(G P^{-1} Qth + Hth)' nu              (nu zero off the active set)

Margin: tightening h_J by delta_J (h -> h - delta) moves r_J to r_J + delta_J, so while the
active set and the sign of nu are unchanged the price becomes
  1/2 (r_J + delta_J)' S_J^{-1} (r_J + delta_J) = Delta + nu_J' delta_J + 1/2 delta_J' S_J^{-1} delta_J.
"""

from __future__ import annotations

import numpy as np

from .solver import ExactQP, ExactResult


def value_gradient(res: ExactResult, Qth: np.ndarray, Hth: np.ndarray) -> np.ndarray:
    """dV/dtheta for every sample, shape (m, B). QP part of the cost (excludes ds/dtheta)."""
    return Qth.T @ res.u - Hth.T @ res.nu


def price_gradient(qp: ExactQP, res: ExactResult, Qth: np.ndarray, Hth: np.ndarray) -> np.ndarray:
    """dDelta/dtheta for every sample, shape (m, B)."""
    return -(qp.G @ qp.Pinv @ Qth + Hth).T @ res.nu


def margin_price(qp: ExactQP, res: ExactResult, delta: np.ndarray) -> np.ndarray:
    """Extra cost nu_J' delta_J + 1/2 delta_J' S_J^{-1} delta_J of tightening h by delta (kappa, B).

    Valid only while the active set and the sign of nu do not change (T4). Entries of delta
    outside the active set of a sample are ignored.
    """
    delta = np.asarray(delta, float)
    if delta.ndim == 1:
        delta = delta[:, None]
    B = res.u.shape[1]
    delta = np.repeat(delta, B, axis=1) if delta.shape[1] == 1 else delta
    out = np.full(B, np.nan)
    for b in range(B):
        if not res.found[b]:
            continue
        J = np.where(res.active[:, b])[0]
        if len(J) == 0:
            out[b] = 0.0
            continue
        S = qp.T[np.ix_(J, J)]
        d = delta[J, b]
        out[b] = res.nu[J, b] @ d + 0.5 * d @ np.linalg.solve(S, d)
    return out
