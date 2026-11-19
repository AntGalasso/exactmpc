import numpy as np

from exactmpc.exact import ExactQP, margin_price, price_gradient, value_gradient
from conftest import random_qp


def make(rng, n=6, k=4, m=3):
    P, q0, G, h0 = random_qp(rng, n, k)
    Q = rng.standard_normal((n, m))
    H = rng.standard_normal((k, m)) * 0.5
    return ExactQP(P, G), q0, Q, h0, H


def solve_at(qp, q0, Q, h0, H, theta):
    return qp.solve(q0[:, None] + Q @ theta, h0[:, None] + H @ theta)


def test_price_gradient_matches_finite_differences(rng):
    for _ in range(20):
        qp, q0, Q, h0, H = make(rng)
        theta = rng.standard_normal((3, 1))
        res = solve_at(qp, q0, Q, h0, H, theta)
        g = price_gradient(qp, res, Q, H)[:, 0]
        e = 1e-6
        fd = np.array([(solve_at(qp, q0, Q, h0, H, theta + e * np.eye(3)[:, [i]]).price[0]
                        - solve_at(qp, q0, Q, h0, H, theta - e * np.eye(3)[:, [i]]).price[0]) / (2 * e) for i in range(3)])
        np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)


def test_value_gradient_matches_finite_differences(rng):
    qp, q0, Q, h0, H = make(rng)
    theta = rng.standard_normal((3, 1))
    V = lambda th: (lambda r: 0.5 * r.u[:, 0] @ qp.P @ r.u[:, 0] + (q0 + Q @ th[:, 0]) @ r.u[:, 0])(solve_at(qp, q0, Q, h0, H, th))
    res = solve_at(qp, q0, Q, h0, H, theta)
    g = value_gradient(res, Q, H)[:, 0]
    e = 1e-6
    fd = np.array([(V(theta + e * np.eye(3)[:, [i]]) - V(theta - e * np.eye(3)[:, [i]])) / (2 * e) for i in range(3)])
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)


def test_price_gradient_is_continuous_across_active_set_changes(rng):
    """T1, numerical evidence: dDelta/dtheta does not jump at active-set changes, du/dtheta does."""
    jumps_price, jumps_u, switches = [], [], 0
    for _ in range(6):
        qp, q0, Q, h0, H = make(rng)
        theta = rng.standard_normal(3)
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        at = lambda t: solve_at(qp, q0, Q, h0, H, (theta + t * d)[:, None])
        ts = np.linspace(-2, 2, 161)
        masks = [tuple(at(t).active[:, 0]) for t in ts]
        for i in range(1, len(ts)):
            if masks[i] == masks[i - 1]:
                continue
            a, b = ts[i - 1], ts[i]
            for _ in range(60):
                c = 0.5 * (a + b)
                a, b = (c, b) if tuple(at(c).active[:, 0]) == masks[i - 1] else (a, c)
            lo, hi = at(a - 1e-9), at(b + 1e-9)
            gl = (price_gradient(qp, lo, Q, H)[:, 0] @ d)
            gh = (price_gradient(qp, hi, Q, H)[:, 0] @ d)
            e = 1e-7
            sl = (at(a - 1e-9).u - at(a - 1e-9 - e).u)[:, 0] / e
            sh = (at(b + 1e-9 + e).u - at(b + 1e-9).u)[:, 0] / e
            jumps_price.append(abs(gl - gh))
            jumps_u.append(np.linalg.norm(sl - sh))
            switches += 1
    assert switches >= 5
    assert max(jumps_price) < 1e-5
    assert np.median(jumps_u) > 1e-2  # the optimiser itself is only continuous


def test_margin_price_matches_resolve(rng):
    checked = 0
    for _ in range(60):
        qp, q0, Q, h0, H = make(rng)
        theta = rng.standard_normal((3, 1))
        res = solve_at(qp, q0, Q, h0, H, theta)
        if not res.active.any():
            continue
        delta = np.where(res.active[:, 0], 1e-3, 0.0)
        res2 = qp.solve(q0[:, None] + Q @ theta, (h0[:, None] + H @ theta) - delta[:, None])
        if not np.array_equal(res.active, res2.active):
            continue
        extra = margin_price(qp, res, delta)[0]
        np.testing.assert_allclose(res2.price[0] - res.price[0], extra, rtol=1e-6, atol=1e-10)
        checked += 1
    assert checked >= 20
