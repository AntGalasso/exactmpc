"""Singular Hessian (P >= 0): over-actuated axes, zero input weight (contribution C4).

STATUS: STUB. Raises NotImplementedError. Covered by an expected-failure test.

What has to be built (statement T3, hypotheses H4 and H5):

1. Range test. With P >= 0, the unconstrained problem has a minimiser iff q is in range(P);
   the minimiser set is u_p + null(P). Use the pseudo-inverse P^+ for the particular solution.
2. Equalities (verified numerically in earlier work, not yet in this repo): active constraints
   C u = d are absorbed at zero cost iff  d - C u_p  is in  range(C V_N),  V_N a basis of null(P).
   Beyond capacity the price is  r_perp' (B Sigma^-2 B')^+ r_perp  (see the proposal, R2).
3. Inequalities on the optimal face {u_p + V_N z}: feasibility has to be tested over the whole
   face, not only at the minimum-norm point (range tests of Lin and Liang, Algorithm 3.1).
   OPEN QUESTION: whether this keeps the 2^kappa bound or needs a second-level enumeration
   over z. Gate G2 of the work plan decides this.
4. Select the minimum-norm minimiser so that the control law is single-valued.

Reference to compare against: warm-started OSQP/DAQP/qpOASES on the dual-drive plant (H5).
"""

from __future__ import annotations

import numpy as np


def solve_exact_singular(P: np.ndarray, q: np.ndarray, G: np.ndarray, h: np.ndarray):
    raise NotImplementedError("P >= 0 with inequalities: see module docstring (gate G2, hypothesis H5)")
