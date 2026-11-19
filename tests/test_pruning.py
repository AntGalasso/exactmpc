import numpy as np

from exactmpc.exact import ExactQP, solve_exact_pruned
from conftest import random_qp


def test_pruned_equals_full_enumeration(rng):
    visited_total, subsets_total = 0, 0
    for _ in range(150):
        P, q, G, h = random_qp(rng, n=8, k=7)
        full = ExactQP(P, G).solve(q, h)
        u, nu, J, visited, total = solve_exact_pruned(P, q, G, h)
        assert full.found[0] and u is not None
        np.testing.assert_allclose(u, full.u[:, 0], atol=1e-8)
        assert visited <= total
        visited_total += visited
        subsets_total += total
    # efficiency is NOT claimed (H6); only report that pruning never visits more than the full set
    assert visited_total <= subsets_total
