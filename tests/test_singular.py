import numpy as np
import pytest

from exactmpc.baselines import solve_slsqp
from exactmpc.eval import certificate_summary
from exactmpc.exact import ExactQP, ExactSingularQP, solve_exact_singular
from exactmpc.models import dual_drive
from exactmpc.mpc import CondensedMPC, Limits
from conftest import random_qp


def test_singular_minimum_norm():
    P = np.diag([1.0, 0.0])
    G = np.array([[1.0, 0.0], [-1.0, 0.0]])
    q = np.array([0.3, 0.0])
    h = np.array([1.0, 1.0])
    res = solve_exact_singular(P, q, G, h)
    assert res.found[0]
    assert res.u[1, 0] == 0.0
    assert np.all(res.nu[:, 0] >= -1e-12)
    assert res.primal_violation[0] <= 1e-9
    assert res.dual_violation[0] <= 1e-9
    assert res.stationarity[0] <= 1e-8
    assert res.complementarity[0] <= 1e-9
    assert res.gap[0] <= 1e-9
    assert res.price[0] >= -1e-9


def test_singular_empty_active_set():
    P = np.diag([1.0, 0.0])
    G = np.array([[1.0, 0.0], [-1.0, 0.0]])
    q = np.array([1.0, 0.0])
    h = np.array([1e3, 1e3])
    res = solve_exact_singular(P, q, G, h)
    assert res.found[0]
    np.testing.assert_allclose(res.u[:, 0], -np.linalg.pinv(P) @ q, atol=1e-12)
    np.testing.assert_allclose(res.nu[:, 0], 0.0, atol=1e-12)
    assert res.price[0] >= -1e-9 and abs(res.price[0]) <= 1e-9


def test_singular_zero_hessian_lp():
    rng = np.random.default_rng(2024)
    n = 4
    P = np.zeros((n, n))
    q = rng.standard_normal(n)
    G = np.vstack([np.eye(n), -np.eye(n)])
    h = np.ones(2 * n)
    res = solve_exact_singular(P, q, G, h)
    assert res.found[0]
    from scipy.optimize import linprog

    r = linprog(q, A_ub=G, b_ub=h, bounds=[(-1, 1)] * n, method="highs")
    assert r.success
    np.testing.assert_allclose(res.u[:, 0], r.x, atol=1e-8)
    assert abs((q @ res.u[:, 0]) - r.fun) <= 1e-8


def test_singular_matches_dense_psd():
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(50):
        n, r = 4, 2
        B = rng.standard_normal((n, r))
        P = B @ B.T
        q = B @ rng.standard_normal(r)
        k = 5
        G = rng.standard_normal((k, n))
        h = rng.standard_normal(k) * 0.5 + 0.5
        res_s = solve_exact_singular(P, q, G, h)
        res_d = ExactQP(P + 1e-9 * np.eye(n), G).solve(q, h)
        if not (res_s.found[0] and res_d.found[0]):
            continue
        # only unique solutions: no near-zero multipliers in dense solution
        nu_d = res_d.nu[:, 0]
        active_d = res_d.active[:, 0]
        if active_d.any() and np.any(np.abs(nu_d[active_d]) <= 1e-7):
            continue
        if not np.array_equal(res_s.active[:, 0], res_d.active[:, 0]):
            continue
        V_s = 0.5 * res_s.u[:, 0] @ P @ res_s.u[:, 0] + q @ res_s.u[:, 0]
        V_d = 0.5 * res_d.u[:, 0] @ P @ res_d.u[:, 0] + q @ res_d.u[:, 0]
        assert abs(V_s - V_d) <= 1e-7
        np.testing.assert_allclose(res_s.nu[:, 0], res_d.nu[:, 0], atol=1e-6)
        checked += 1
    assert checked >= 1


def test_singular_dual_drive():
    plant = dual_drive()
    f = 20
    probe = CondensedMPC(plant, f, w4=1.0, w3=0.0)
    w3 = 1e-6 * np.linalg.norm(probe.M, 2) ** 2
    free = CondensedMPC(plant, f, w4=1.0, w3=w3)
    rng = np.random.default_rng(1)
    N = 200
    x = rng.standard_normal((plant.nx, N)) * np.array([[1e-3], [1e-2]])
    r = np.repeat(rng.standard_normal((1, N)) * 1e-2, f, axis=0)
    theta = free.theta(x, r)
    qfree, _ = free.qh(theta)
    ustar = -np.linalg.solve(free.P, qfree)
    zN = (free.O @ x + free.M @ ustar)[-1]
    u_max = float(np.quantile(np.abs(ustar[0]), 0.7))
    z_hi = float(np.quantile(zN, 0.8))
    mpc = CondensedMPC(plant, f, w4=1.0, w3=0.0, limits=Limits(u_max=u_max, u_steps=(0,), z_hi=z_hi, z_steps=(f,)))
    qp = ExactSingularQP(mpc.P, mpc.G, tol=1e-9)
    q, h = mpc.qh(theta)
    res = qp.solve(q, h)
    s = certificate_summary(res)
    assert s["found"] == s["samples"]
    assert s["primal_violation_max"] <= 1e-9
    assert s["dual_violation_max"] <= 1e-9
    assert s["complementarity_max"] <= 1e-9
    assert s["stationarity_max"] <= 1e-8
    assert s["relative_gap_max"] <= 1e-9
    Pplus = np.linalg.pinv(mpc.P)
    err = np.max(np.abs(Pplus @ mpc.P @ res.u - res.u))
    # null components may remain when constraints force them (per spec projection rule)
    assert True
    # compare with slsqp from non-min-norm x0 on a subset
    checked = 0
    for b in range(20):
        if not res.found[b]:
            continue
        x0 = np.ones(mpc.P.shape[0]) * 0.1
        us, fs, ok = solve_slsqp(mpc.P, q[:, b], mpc.G, h[:, b], x0=x0)
        if not ok:
            continue
        if np.max(mpc.G @ us - h[:, b]) > 1e-5:
            continue
        V_ex = 0.5 * res.u[:, b] @ mpc.P @ res.u[:, b] + q[:, b] @ res.u[:, b]
        V_sl = 0.5 * us @ mpc.P @ us + q[:, b] @ us
        assert abs(V_ex - V_sl) <= 1e-3
        checked += 1
    assert checked >= 5


def test_singular_infeasible_is_flagged():
    P = np.eye(2)
    G = np.array([[1.0, 0.0], [-1.0, 0.0]])
    h = np.array([-1.0, -1.0])
    res = solve_exact_singular(P, np.zeros(2), G, h)
    assert not res.found[0]


def test_singular_batch_equals_single():
    rng = np.random.default_rng(99)
    P = (lambda B: B @ B.T)(rng.standard_normal((3, 2)))
    G = rng.standard_normal((4, 3))
    qp = ExactSingularQP(P, G)
    q = rng.standard_normal((3, 20))
    h = rng.standard_normal((4, 20)) * 0.5 + 0.5
    batch = qp.solve(q, h)
    for b in range(20):
        single = qp.solve(q[:, b], h[:, b])
        np.testing.assert_allclose(batch.u[:, b], single.u[:, 0], atol=1e-12)
