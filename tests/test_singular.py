import numpy as np
import pytest

from exactmpc.exact.singular import solve_exact_singular


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="T3 / gate G2: singular Hessian with inequalities is not implemented")
def test_singular_hessian_minimum_norm():
    solve_exact_singular(np.zeros((2, 2)), np.zeros(2), np.eye(2), np.ones(2))
