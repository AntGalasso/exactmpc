"""Metrics.

STATUS: certificate_summary IMPLEMENTED. Contour error, cycle time and violation counters are
STUBS to be added with the planner (hypotheses H3, H4).
"""

from __future__ import annotations

import numpy as np

from ..exact.solver import ExactResult


def certificate_summary(res: ExactResult) -> dict:
    """Worst-case certificate residuals over the batch (H1 thresholds: all <= 1e-9)."""
    ok = res.found
    f = lambda a: float(np.nanmax(a[ok])) if ok.any() else float("nan")
    return {
        "samples": int(ok.size),
        "found": int(ok.sum()),
        "primal_violation_max": f(res.primal_violation),
        "dual_violation_max": f(res.dual_violation),
        "complementarity_max": f(res.complementarity),
        "stationarity_max": f(res.stationarity),
        "relative_gap_max": f(res.gap),
        "distinct_active_sets": len({tuple(np.where(res.active[:, b])[0]) for b in np.where(ok)[0]}),
    }
