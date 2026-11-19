"""exactmpc: exact constrained MPC with a certified closed-form layer.

Layout
------
models/      plants (feed drive, dual-drive axis, double integrator)
mpc/         condensed MPC as a parametric QP, and the limits that define G, h(theta)
exact/       the exact layer: enumeration, certificate, price, margin, pruning, singular P
interface/   closed-form objects exported to the planner: no-limit polyhedron A0, feasible set F
planner/     learned planner above the exact layer (stubs, needs torch)
baselines/   reference and competing solvers/planners
eval/        metrics

Every module starts with a STATUS line: IMPLEMENTED (and which test covers it) or STUB.
"""

__version__ = "0.0.1"
