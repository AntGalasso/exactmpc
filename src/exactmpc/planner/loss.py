"""Planner loss.

STATUS: STUB.

loss = contour_error + cycle_time_term + lambda * Delta (exact price)
The price is a SOFT penalty: the planner may leave the no-limit polyhedron A0 when it pays.
Feasibility is guaranteed by the guard set, not by the penalty.
Weights and the cycle-time term must be fixed with K. Tsabedze before any experiment (H3).
"""


def planner_loss(*args, **kwargs):
    raise NotImplementedError("see module docstring")
