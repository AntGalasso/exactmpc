"""Limits that define the constraint rows G u <= h(theta).

STATUS: IMPLEMENTED (covered by tests/test_condense.py and tests/test_feasible_set.py).

Design rule from the project: the number of inequality rows kappa must stay small
(kappa <= 6, so at most 63 candidates). Limits are therefore applied only at the steps
listed in `*_steps`, not at every step of the horizon. Bounds on every step are possible
(they are used in the feasibility checks) but make the enumeration 2^kappa non viable.

Only single-output plants (ny = 1) are supported in this skeleton.

Parameter theta = [x; r] with x the measured state and r the reference over the horizon.
Constraint rows are affine in theta: h(theta) = h0 + Hth theta.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Limits:
    # |u_k| <= u_max for k in u_steps (0-based, step 0 is the input applied now)
    u_max: float | None = None
    u_steps: tuple[int, ...] = (0,)
    # absolute bounds on the predicted output z_k, k in z_steps (1-based as in z_1..z_f)
    z_lo: float | None = None
    z_hi: float | None = None
    z_steps: tuple[int, ...] = ()
    # contour-error style band |z_k - r_k| <= band for k in band_steps (1-based).
    # THIS CONSTRAINT DEPENDS ON THE REFERENCE and can make the QP infeasible.
    band: float | None = None
    band_steps: tuple[int, ...] = ()
    names: list[str] = field(default_factory=list)
