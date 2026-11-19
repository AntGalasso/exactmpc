import numpy as np

from exactmpc.interface import max_feasible_scale, qp_feasible
from exactmpc.models import double_integrator
from exactmpc.mpc import CondensedMPC, Limits


def test_input_limits_alone_are_always_feasible(rng):
    f = 10
    mpc = CondensedMPC(double_integrator(), f=f, limits=Limits(u_max=1.0, u_steps=tuple(range(f))))
    for _ in range(20):
        theta = mpc.theta(rng.standard_normal(2) * 5, rng.standard_normal(f) * 50)
        _, h = mpc.qh(theta)
        assert qp_feasible(mpc.G, h[:, 0])


def test_reference_dependent_band_can_make_the_qp_infeasible():
    f = 10
    lim = Limits(u_max=1.0, u_steps=tuple(range(f)), band=0.05, band_steps=tuple(range(1, f + 1)))
    mpc = CondensedMPC(double_integrator(), f=f, limits=lim)
    theta0 = mpc.theta(np.zeros(2), 0.0)[:, 0]
    direction = np.concatenate([np.zeros(2), np.ones(f)])
    t_star = max_feasible_scale(mpc.G, mpc.h0, mpc.Hth, theta0, direction)
    assert 0.054 < t_star < 0.056  # step reference from rest: infeasible beyond about 0.055
    _, h = mpc.qh(mpc.theta(np.zeros(2), 1.0))
    assert not qp_feasible(mpc.G, h[:, 0])
