"""Plants.

STATUS: IMPLEMENTED (covered by tests/test_condense.py).

The feed-drive parameters below are PLACEHOLDERS (unit mass, light damping). They must be
replaced with the CNC model of the lab (to be provided by K. Tsabedze) before any result is
reported. Nothing in this file is a measured or published value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import cont2discrete


@dataclass(frozen=True)
class Plant:
    """Discrete-time LTI plant x+ = A x + B u, y = C x."""

    name: str
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Ts: float

    @property
    def nx(self) -> int:
        return self.A.shape[0]

    @property
    def nu(self) -> int:
        return self.B.shape[1]

    @property
    def ny(self) -> int:
        return self.C.shape[0]


def _zoh(Ac: np.ndarray, Bc: np.ndarray, Ts: float) -> tuple[np.ndarray, np.ndarray]:
    n, m = Bc.shape
    Ad, Bd, _, _, _ = cont2discrete((Ac, Bc, np.eye(n), np.zeros((n, m))), Ts, method="zoh")
    return Ad, Bd


def feed_drive(mass: float = 1.0, damping: float = 0.5, gain: float = 1.0, Ts: float = 1e-3) -> Plant:
    """Single-axis second-order feed drive, state [position; velocity], input force/current.

    PLACEHOLDER parameters, see module docstring.
    """
    Ac = np.array([[0.0, 1.0], [0.0, -damping / mass]])
    Bc = np.array([[0.0], [gain / mass]])
    Ad, Bd = _zoh(Ac, Bc, Ts)
    return Plant("feed_drive", Ad, Bd, np.array([[1.0, 0.0]]), Ts)


def dual_drive(mass: float = 1.0, damping: float = 0.5, gains: tuple[float, float] = (1.0, 1.0), Ts: float = 1e-3) -> Plant:
    """Dual-drive axis: two actuators act on one axis (two inputs, one output).

    The two input columns of the prediction matrix are collinear, so with zero input
    weight the condensed Hessian is singular (target of contribution C4 / hypothesis H5).
    PLACEHOLDER parameters.
    """
    Ac = np.array([[0.0, 1.0], [0.0, -damping / mass]])
    Bc = np.array([[0.0, 0.0], [gains[0] / mass, gains[1] / mass]])
    Ad, Bd = _zoh(Ac, Bc, Ts)
    return Plant("dual_drive", Ad, Bd, np.array([[1.0, 0.0]]), Ts)


def double_integrator(Ts: float = 0.1) -> Plant:
    """Double integrator used in the feasibility checks (position output)."""
    A = np.array([[1.0, Ts], [0.0, 1.0]])
    B = np.array([[0.5 * Ts**2], [Ts]])
    return Plant("double_integrator", A, B, np.array([[1.0, 0.0]]), Ts)
