# Work packages and gates (weeks counted from the kick-off, a proposal to agree on)

| WP | Content | Weeks | Hypotheses | Gate | Code |
| --- | --- | --- | --- | --- | --- |
| 0 | Alignment: scope, plants, thresholds, roles; prior-art check on control allocation, OptNet, convex TOPP, tracking MPC | 1 | | | `docs/` |
| 1 | Exact layer on plant A with certificates; baselines OSQP, DAQP, qpOASES | 2-5 | H1, H6 | G1: certificates hold, enumeration cost acceptable | `exact/`, `baselines/qp_solvers.py` |
| 2 | Singular Hessian with inequalities; dual-drive plant B | 4-8 | H5 | G2: go or no-go on T3 for inequalities | `exact/singular.py` |
| 3 | Interface: price, A0, guard set F, price of margin; proofs of T1, T2, T4 | 5-9 | H4 (design) | | `interface/`, `exact/price.py` |
| 4 | Planner; OptNet-style comparison; convex TOPP and jerk-limited baselines; Hose baselines | 9-14 | H2, H3, H4 | G3: planner beats scheduling, or the thesis becomes analytical | `planner/`, `baselines/` |
| 5 | Capability maps and the robustness-performance frontier | 13-16 | H7 | | `experiments/h7_*` |
| 6 | Writing: P1 from week 8, P2 from week 14, thesis chapters in parallel | 8-20 | | | |

If G2 fails, C4 is reduced to equalities. If G3 fails, P2 is dropped and P1 plus the interface results (C2, C5) form the thesis.
