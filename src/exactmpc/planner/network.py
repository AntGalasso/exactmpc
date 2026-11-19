"""Planner network.

STATUS: STUB.

Inputs : local path geometry (curvature, corner angle, segment ahead), measured state x_k,
         optionally the multipliers nu of the previous sample (ablation).
Outputs: a feedrate profile and a small path correction, i.e. the reference r over the horizon.
Constraint handling: a BOUNDED parametrisation (e.g. tanh scaled by amplitude and rate limits
precomputed from the guard set R_adm inside F). Do not project onto a polyhedron with a QP:
its enumeration would be exponential in the number of facets.
"""


class PlannerNet:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("see module docstring")
