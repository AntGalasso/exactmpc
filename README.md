# exactmpc

Project skeleton for the thesis *Certified Learning Above Exact Control: Price-Coordinated Planning for Constrained MPC of CNC Feed Drives*.

Thesis in one line: a constrained MPC servo solved exactly (closed form, KKT certificate per sample, no tolerance) is a certified, differentiable interface to a learned planner, so that guarantees come from the structure of the interface and not from verifying the network.

```
planner network --reference r--> exact MPC layer --input u--> feed drive
       ^                               |
       +---- price, multipliers, A0 ---+      (closed form, exact gradient)
```

## What works today and what does not

| Part | Status |
| --- | --- |
| `models/` plants | Implemented. Feed-drive parameters are PLACEHOLDERS, not the lab model. |
| `mpc/` condensed MPC and limits | Implemented, tested against simulation. |
| `exact/solver.py` enumeration with KKT certificate (P > 0) | Implemented, tested against SLSQP and certificate thresholds. |
| `exact/price.py` price, gradient, price of margin | Implemented, tested against finite differences and a re-solve. |
| `exact/pruning.py` sound pruning (single QP) | Implemented, soundness tested; efficiency NOT established. |
| `interface/` polyhedron A0, feasibility checks | Implemented, tested. |
| `exact/singular.py` P >= 0 with inequalities | STUB (contribution C4, gate G2). |
| `interface/feasible_set.GuardSet` | STUB (statement T2). |
| `planner/`, `baselines/` (except SLSQP reference) | STUBS. Need torch, OSQP, DAQP. |
| `experiments/h1_certificates.py` | Runs on the placeholder plant. H2 to H7 are stubs. |

Nothing in this repository is a result of the project yet. The H1 script prints numbers for a placeholder plant; they only show that the pipeline runs.

## Run

```bash
pip install -e ".[dev]"      # numpy, scipy, pytest
make test                    # 18 passed, 1 expected failure (singular Hessian stub)
make h1                      # certificate pipeline on the placeholder feed drive
```

## Layout

```
src/exactmpc/
  models/      plants
  mpc/         condensed MPC, limits (kappa kept small by design)
  exact/       solver, price, pruning, singular (stub)
  interface/   no-limit polyhedron A0, feasible set F, guard set (stub)
  planner/     network, differentiable layer, loss (stubs)
  baselines/   SLSQP reference; OSQP/DAQP, TOPP, OptNet, Hose (stubs)
  eval/        certificate metrics
experiments/   h1 (runs), h2..h7 (stubs)
tests/         one file per implemented module
docs/          hypotheses.md (thresholds, kill criteria), theory_status.md (T1..T6), roadmap.md
```

## Conventions

- `theta = [x; r]`: measured state and reference over the horizon. `q = Qth theta`, `h = h0 + Hth theta`.
- Samples are columns: `q` is `(n, B)`, `h` is `(kappa, B)`.
- Every module starts with a `STATUS:` line saying what is implemented, tested or a stub.
- Keep kappa <= 6 (at most 63 candidates). Bounds on every step of the horizon make the enumeration non viable.

## Next steps

1. Replace the placeholder plant with the lab's CNC model and fix the limits and the contour-error band (hard or soft) with K. Tsabedze.
2. Gate G2: decide whether the singular Hessian with inequalities admits a closed form (`exact/singular.py`).
3. Prove T1 and T2 on paper; the code only gives numerical evidence for T1.
4. Install torch and implement `planner/layer.py` with a finite-difference gradient check before any training.
