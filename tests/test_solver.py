import numpy as np
import pytest

from exactmpc.baselines import solve_slsqp
from exactmpc.eval import certificate_summary
from exactmpc.exact import ExactQP
from conftest import random_qp


def test_matches_slsqp_and_certificates_hold(rng):
    n_found = 0
    for _ in range(100):
        P, q, G, h = random_qp(rng, n=6, k=4)
        qp = ExactQP(P, G)
        res = qp.solve(q, h)
        assert res.found[0]
        x_ref, f_ref, ok = solve_slsqp(P, q, G, h)
        f_exact = 0.5 * res.u[:, 0] @ P @ res.u[:, 0] + q @ res.u[:, 0]
        assert f_exact <= f_ref + 1e-7
        np.testing.assert_allclose(res.u[:, 0], x_ref, atol=1e-4)
        n_found += 1
    assert n_found == 100


def test_certificate_thresholds_h1(rng):
    P, _, G, _ = random_qp(rng, n=6, k=4)
    qp = ExactQP(P, G)
    q = rng.standard_normal((6, 500)) * 3
    h = rng.standard_normal((4, 500)) * 0.5 + 0.5
    res = qp.solve(q, h)
    s = certificate_summary(res)
    assert s["found"] == s["samples"]
    for key in ("primal_violation_max", "dual_violation_max", "complementarity_max", "relative_gap_max"):
        assert s[key] <= 1e-9, (key, s[key])
    assert s["stationarity_max"] <= 1e-8


def test_batch_equals_single(rng):
    P, _, G, _ = random_qp(rng)
    qp = ExactQP(P, G)
    q = rng.standard_normal((6, 20))
    h = rng.standard_normal((4, 20)) * 0.5 + 0.5
    batch = qp.solve(q, h)
    for b in range(20):
        single = qp.solve(q[:, b], h[:, b])
        np.testing.assert_allclose(batch.u[:, b], single.u[:, 0], atol=1e-12)


def test_infeasible_qp_is_flagged(rng):
    P = np.eye(2)
    G = np.array([[1.0, 0.0], [-1.0, 0.0]])
    h = np.array([-1.0, -1.0])  # x <= -1 and x >= 1
    res = ExactQP(P, G).solve(np.zeros(2), h)
    assert not res.found[0]


def test_rejects_singular_hessian():
    with pytest.raises(ValueError):
        ExactQP(np.zeros((2, 2)), np.eye(2))


def test_rows_of_very_different_scale_are_not_declared_dependent(rng):
    """Regression: an input-limit row (scale 1) and an output-limit row (scale 1e-6) were wrongly dropped."""
    P, q, G, h = random_qp(rng, n=6, k=3)
    scale = np.array([1.0, 1e-6, 1.0])
    qp_scaled = ExactQP(P, G * scale[:, None])
    qp_plain = ExactQP(P, G)
    assert len(qp_scaled.subsets) == len(qp_plain.subsets)
    for _ in range(30):
        q = rng.standard_normal(6) * 3
        h = rng.standard_normal(3) * 0.5 + 0.5
        a = qp_plain.solve(q, h)
        b = qp_scaled.solve(q, h * scale)
        assert a.found[0] and b.found[0]
        np.testing.assert_allclose(a.u, b.u, atol=1e-7)
        np.testing.assert_allclose(a.price, b.price, rtol=1e-7, atol=1e-12)
