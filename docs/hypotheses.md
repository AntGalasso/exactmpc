# Hypotheses and kill criteria

Thresholds are fixed before any run. A hypothesis that fails changes the thesis; it is not explained away.

| ID | Hypothesis | Threshold or test | If falsified | Code |
| --- | --- | --- | --- | --- |
| H1 | Certificates hold on all instances | Primal violation <= 1e-9, nu >= -1e-9, relative gap <= 1e-9 | Any systematic failure: the certificate claim is withdrawn and the cause analysed | `experiments/h1_certificates.py` (runs), `tests/test_solver.py` |
| H2 | Training through the exact layer is at least as good as an OptNet-style layer | Same or better loss and wall-clock than the layer at tolerance 1e-3 and 1e-6 | If the layer at 1e-6 matches: C2 reduces to determinism and the certificate | `experiments/h2_*` (stub) |
| H3 | The planner beats strong scheduling | Shorter cycle time at equal contour error than jerk-limited and convex TOPP | Within noise: drop the planner claim, thesis is analytical (P1) | `experiments/h3_*` (stub) |
| H4 | Safety by construction | Zero limit violations over randomised references inside the guard set, while the naive network violates | Any violation inside the guard set: T2 corrected or restricted | `experiments/h4_*` (stub) |
| H5 | Singular case | Warm-started solvers return initialisation-dependent null-space components; the exact layer returns the minimum-norm one | Solvers agree, or the difference is immaterial in closed loop: C4 becomes a remark | `experiments/h5_*` (stub) |
| H6 | Pruning | Candidates visited << 2^kappa - 1 | More than half of all subsets occur: the pruning claim is withdrawn | `experiments/h6_*` (stub), `tests/test_pruning.py` (soundness only) |
| H7 | Capability maps | Price and multipliers along a contour match finite-difference re-solves to 1e-6 and identify the binding limit | Disagreement: the map is not reported | `experiments/h7_*` (stub) |

Plants: (A) single-axis feed drive, (B) dual-drive axis (singular Hessian), (C) two-axis contouring. The feed-drive parameters in `models/plants.py` are PLACEHOLDERS until the lab model arrives.
