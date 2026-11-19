import numpy as np

from exactmpc.models import double_integrator, dual_drive, feed_drive
from exactmpc.mpc import CondensedMPC, Limits, condense


def simulate(plant, x0, u_seq):
    x, z = x0.copy(), []
    for u in u_seq:
        x = plant.A @ x + plant.B @ u
        z.append((plant.C @ x).item())
    return np.array(z)


def test_condense_matches_simulation(rng):
    for plant in (double_integrator(), feed_drive(), dual_drive()):
        f = 8
        O, M = condense(plant, f)
        x0 = rng.standard_normal(plant.nx)
        u = rng.standard_normal((f, plant.nu))
        np.testing.assert_allclose(O @ x0 + M @ u.ravel(), simulate(plant, x0, u), atol=1e-12)


def test_qp_cost_matches_tracking_cost(rng):
    plant = double_integrator()
    mpc = CondensedMPC(plant, f=6, w4=2.0, w3=0.3)
    x = rng.standard_normal(2)
    r = rng.standard_normal(6)
    u = rng.standard_normal(6)
    theta = mpc.theta(x, r)
    q, _ = mpc.qh(theta)
    qp_cost = 0.5 * u @ mpc.P @ u + q[:, 0] @ u + mpc.constant(theta)[0]
    e = r - mpc.O @ x - mpc.M @ u
    direct = 2.0 * e @ e + 0.3 * u @ u
    np.testing.assert_allclose(qp_cost, direct, rtol=1e-10)


def test_constraint_rows_match_definitions(rng):
    plant = double_integrator()
    lim = Limits(u_max=1.0, u_steps=(0, 1), z_hi=2.0, z_lo=-1.5, z_steps=(3,), band=0.05, band_steps=(2, 4))
    mpc = CondensedMPC(plant, f=5, limits=lim)
    assert mpc.kappa == 4 + 2 + 4
    x, r, u = rng.standard_normal(2), rng.standard_normal(5), rng.standard_normal(5)
    theta = mpc.theta(x, r)
    _, h = mpc.qh(theta)
    slack = h[:, 0] - mpc.G @ u  # >= 0 means satisfied
    z = mpc.O @ x + mpc.M @ u
    expected = [1 - u[0], 1 + u[0], 1 - u[1], 1 + u[1], 2.0 - z[2], z[2] + 1.5,
                0.05 - (z[1] - r[1]), 0.05 - (r[1] - z[1]), 0.05 - (z[3] - r[3]), 0.05 - (r[3] - z[3])]
    np.testing.assert_allclose(slack, expected, atol=1e-12)


def test_dual_drive_hessian_is_singular_without_input_weight():
    plant = dual_drive()
    mpc = CondensedMPC(plant, f=5, w3=0.0)
    lam = np.linalg.eigvalsh(mpc.P)
    assert lam[0] < 1e-9 * lam[-1]
    assert np.linalg.matrix_rank(mpc.P, tol=1e-9 * lam[-1]) <= 5
