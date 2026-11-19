import numpy as np
import pytest


def random_qp(rng, n=6, k=4, scale_h=0.5):
    X = rng.standard_normal((20, n))
    P = 2 * X.T @ X + 1e-2 * np.eye(n)
    q = rng.standard_normal(n) * 3
    G = rng.standard_normal((k, n))
    h = rng.standard_normal(k) * scale_h + 0.5
    return P, q, G, h


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
