"""OSQP, DAQP and qpOASES wrappers for the servo comparison (hypotheses H1, H5, H6).

STATUS: STUB. Raises NotImplementedError.

Install with `pip install osqp daqp` (optional extra "baselines"). Each wrapper must take (P, q, G, h) and return u, time, iterations. For H5 also expose warm start so the dependence of the null-space component on initialisation can be measured.
"""


def solve_osqp(*args, **kwargs):
    raise NotImplementedError("OSQP, DAQP and qpOASES wrappers for the servo comparison (hypotheses H1, H5, H6).")
