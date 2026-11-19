"""H1: certificates hold on all instances (primal violation, nu >= 0, relative gap <= 1e-9).

STATUS: IMPLEMENTED as a runnable pipeline on the PLACEHOLDER feed drive. The numbers it prints
are not results of the project until the plant is replaced with the lab's CNC model and the
thresholds are confirmed at the kick-off.

Scaling: the input weight is set relative to sigma_max(M)^2 (so cond(P) is controlled by
--w3-rel) and the limits are set at quantiles of the unconstrained solution, so that limits
are active in a sizeable fraction of the samples. Without this the placeholder plant
(tiny prediction matrix at Ts = 1 ms) gives a trivial or badly scaled problem.

Run:  python experiments/h1_certificates.py --samples 2000 --f 20 --w3-rel 1e-6
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from exactmpc.baselines import solve_slsqp
from exactmpc.eval import certificate_summary
from exactmpc.exact import ExactQP
from exactmpc.models import feed_drive
from exactmpc.mpc import CondensedMPC, Limits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--f", type=int, default=20)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--w3-rel", type=float, default=1e-6, help="input weight relative to sigma_max(M)^2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", type=int, default=30, help="samples re-solved with SLSQP")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    plant = feed_drive()
    probe = CondensedMPC(plant, a.f, w4=1.0, w3=0.0)
    w3 = a.w3_rel * np.linalg.norm(probe.M, 2) ** 2
    free = CondensedMPC(plant, a.f, w4=1.0, w3=w3)

    x = rng.standard_normal((plant.nx, a.samples)) * np.array([[1e-3], [1e-2]])
    r = np.repeat(rng.standard_normal((1, a.samples)) * 1e-2, a.f, axis=0)
    theta = free.theta(x, r)
    q, _ = free.qh(theta)
    ustar = -np.linalg.solve(free.P, q)
    zN = (free.O @ x + free.M @ ustar)[-1]
    u_max = float(np.quantile(np.abs(ustar[0]), 0.7))
    z_hi = float(np.quantile(zN, 0.8))

    # kappa = 3: input limits now (2 rows) and one output limit at the end of the horizon
    mpc = CondensedMPC(plant, a.f, w4=1.0, w3=w3, limits=Limits(u_max=u_max, u_steps=(0,), z_hi=z_hi, z_steps=(a.f,)))
    qp = ExactQP(mpc.P, mpc.G, tol=a.tol)
    q, h = mpc.qh(theta)
    print(f"plant={plant.name} (placeholder) f={a.f} n={mpc.n} kappa={mpc.kappa} candidates={qp.n_candidates} cond(P)={np.linalg.cond(mpc.P):.2e}")

    t0 = time.perf_counter()
    res = qp.solve(q, h)
    dt = time.perf_counter() - t0
    s = certificate_summary(res)
    print(f"solved {a.samples} states in {1e3 * dt:.1f} ms ({1e6 * dt / a.samples:.1f} us/state, one CPU core, Python/NumPy)")
    for k, v in s.items():
        print(f"  {k}: {v}")
    print(f"  samples with at least one active limit: {int(res.active.any(axis=0).sum())}")

    rel = []
    for b in range(min(a.check, a.samples)):
        if not res.found[b]:
            continue
        x_ref, f_ref, _ = solve_slsqp(mpc.P, q[:, b], mpc.G, h[:, b])
        f_ex = 0.5 * res.u[:, b] @ mpc.P @ res.u[:, b] + q[:, b] @ res.u[:, b]
        rel.append((f_ex - f_ref) / max(1e-300, abs(f_ref)))
    if rel:
        print(f"  (V_exact - V_SLSQP)/|V_SLSQP| over {len(rel)} samples: max {max(rel):.2e}, min {min(rel):.2e} (negative means exact is lower)")
    h1_ok = s["found"] == s["samples"] and all(s[k] <= 1e-9 for k in ("primal_violation_max", "dual_violation_max", "relative_gap_max"))
    print("H1 threshold met on this run:", h1_ok)


if __name__ == "__main__":
    main()
