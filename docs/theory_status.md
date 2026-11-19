# Theory status

Notation: P > 0 Hessian, G_J rows of the active set J, S_J = G_J P^-1 G_J', u* = -P^-1 q, r_J = G_J u* - h_J, theta = [x; r] enters q and h affinely.

| ID | Statement | Status | Where |
| --- | --- | --- | --- |
| T1 | Value function V(theta) is C^1 (envelope gradient); price Delta = V - V_unc is C^1 and piecewise quadratic; the optimal input is only continuous and piecewise affine | Numerical evidence only (gradient jump <= 1e-5 across active-set changes, finite-difference agreement). Proof open (continuity of multipliers under LICQ). | `exact/price.py`, `tests/test_price.py` |
| T2 | Feasible reference set F and a guard set R_adm inside it give recursive feasibility under reference changes (reference-generic terminal ingredients) | Open. Hand-check: input limits alone are always feasible; a reference-dependent band is not (`tests/test_feasible_set.py`). | `interface/feasible_set.py` (GuardSet is a stub) |
| T3 | Singular Hessian: unique constrained minimiser conditions, minimum-norm selection, capacity | Equalities checked numerically outside this repo; inequalities on the optimal face open. Gate G2. | `exact/singular.py` (stub) |
| T4 | Price of margin: extra cost nu_J' delta_J + 1/2 delta_J' S_J^-1 delta_J while the active set is unchanged | Algebra checked; tested against a re-solve. Range of validity must be stated in the paper. | `exact/price.py`, `tests/test_price.py` |
| T5 | Clipping: constrained = clipped when P is diagonal and limits are boxes; exact cost of clipping through collinearity otherwise | Diagonal case immediate; general expression open | not in code |
| T6 | Pruning: once Delta_J exceeds the best primal feasible price, every superset of J can be discarded | Soundness argument in `exact/pruning.py` and tested against full enumeration; efficiency is empirical (H6) | `exact/pruning.py`, `tests/test_pruning.py` |

Known caveats of the implementation
- The exact solver assumes P > 0 and linearly independent active rows (LICQ). `found == False` means infeasible, degenerate, or `max_active` too small.
- The enumeration costs 2^kappa - 1 candidates. Keep kappa <= 6 (see `mpc/limits.py`).
- Selecting the lowest-price primal feasible candidate already returns the optimum; the nu >= 0 test keeps every returned point a KKT point.
