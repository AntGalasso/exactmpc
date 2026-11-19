import numpy as np

from exactmpc.exact import ExactQP
from exactmpc.interface import in_polyhedron, no_limit_polyhedron
from exactmpc.models import double_integrator
from exactmpc.mpc import CondensedMPC, Limits


def test_polyhedron_membership_iff_no_active_limit(rng):
    mpc = CondensedMPC(double_integrator(), f=6, w4=1.0, w3=1e-2, limits=Limits(u_max=2.0, u_steps=(0,), z_hi=1.0, z_steps=(6,)))
    qp = ExactQP(mpc.P, mpc.G)
    A, b = no_limit_polyhedron(qp.Pinv, mpc.G, np.zeros(mpc.n), mpc.Qth, mpc.h0, mpc.Hth)
    theta = np.vstack([rng.standard_normal((2, 400)) * 1.0, np.repeat(rng.standard_normal((1, 400)) * 1.5, 6, axis=0)])
    q, h = mpc.qh(theta)
    res = qp.solve(q, h)
    inside = in_polyhedron(A, b, theta)
    assert res.found.all()
    assert inside.any() and (~inside).any()
    np.testing.assert_array_equal(inside, ~res.active.any(axis=0))
    assert np.all(res.price[inside] == 0.0)
