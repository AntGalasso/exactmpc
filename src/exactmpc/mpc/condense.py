"""Condensed MPC as a parametric QP.

STATUS: IMPLEMENTED (covered by tests/test_condense.py).

Cost:  (r - O x - M u)' W4 (r - O x - M u) + u' W3 u,   u in R^n, n = f * nu
QP:    min 1/2 u'Pu + q'u + s   s.t.  G u <= h,
       P = 2 (M' W4 M + W3),  q = Qth theta,  Qth = [2 M' W4 O, -2 M' W4],
       s = (r - O x)' W4 (r - O x),  theta = [x; r].
Predicted outputs are z_k = C x_k for k = 1..f, so z = O x + M u with
O row i = C A^(i+1) and M block (i, j) = C A^(i-j) B for i >= j.
"""

from __future__ import annotations

import numpy as np

from ..models.plants import Plant
from .limits import Limits


def condense(plant: Plant, f: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (O, M) with z = O x + M u over the horizon f (single output)."""
    if plant.ny != 1:
        raise NotImplementedError("single-output plants only in this skeleton")
    A, B, C = plant.A, plant.B, plant.C
    nu = plant.nu
    O = np.zeros((f, plant.nx))
    M = np.zeros((f, f * nu))
    Ap = np.eye(plant.nx)
    powers = []
    for _ in range(f):
        powers.append(Ap)
        Ap = A @ Ap
    for i in range(f):
        O[i] = (C @ A @ powers[i]).ravel()
        for j in range(i + 1):
            M[i, j * nu : (j + 1) * nu] = (C @ powers[i - j] @ B).ravel()
    return O, M


def _weight(w, n: int) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    return w * np.eye(n) if w.ndim == 0 else w


class CondensedMPC:
    """Tracking MPC condensed into one parametric QP with fixed P and G."""

    def __init__(self, plant: Plant, f: int, w4=1.0, w3=1e-6, limits: Limits | None = None):
        self.plant, self.f = plant, f
        self.nx, self.nu_dim = plant.nx, plant.nu
        self.n = f * plant.nu
        self.O, self.M = condense(plant, f)
        self.W4 = _weight(w4, f)
        self.W3 = _weight(w3, self.n)
        M, O, W4 = self.M, self.O, self.W4
        self.P = 2.0 * (M.T @ W4 @ M + self.W3)
        self.theta_dim = self.nx + f
        self.Qth = np.hstack([2.0 * M.T @ W4 @ O, -2.0 * M.T @ W4])
        self.limits = limits if limits is not None else Limits()
        self.G, self.h0, self.Hth = self._build_constraints(self.limits)
        self.kappa = self.G.shape[0]

    # ---- parameter handling -------------------------------------------------------
    def theta(self, x: np.ndarray, r) -> np.ndarray:
        """Stack theta = [x; r] for one or several samples. x: (nx,) or (nx,B); r: scalar,(f,) or (f,B)."""
        x = np.asarray(x, float)
        if x.ndim == 1:
            x = x[:, None]
        B = x.shape[1]
        r = np.asarray(r, float)
        if r.ndim == 0:
            r = np.full((self.f, B), float(r))
        elif r.ndim == 1:
            r = np.repeat(r[:, None], B, axis=1)
        return np.vstack([x, r])

    def qh(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """q(theta) and h(theta), columns are samples."""
        return self.Qth @ theta, self.h0[:, None] + self.Hth @ theta

    def constant(self, theta: np.ndarray) -> np.ndarray:
        """Constant term s(theta) of the cost (needed to recover the full tracking cost)."""
        x, r = theta[: self.nx], theta[self.nx :]
        e = r - self.O @ x
        return np.einsum("ib,ij,jb->b", e, self.W4, e)

    # ---- constraints --------------------------------------------------------------
    def _build_constraints(self, lim: Limits):
        f, n, nu = self.f, self.n, self.nu_dim
        rows, h0, Hth = [], [], []
        T = self.theta_dim

        def add(g, c0, hx, hr):
            rows.append(g)
            h0.append(c0)
            row = np.zeros(T)
            row[: self.nx] = hx
            row[self.nx :] = hr
            Hth.append(row)

        zero_x = np.zeros(self.nx)
        zero_r = np.zeros(f)
        if lim.u_max is not None:
            for k in lim.u_steps:
                for a in range(nu):
                    e = np.zeros(n)
                    e[k * nu + a] = 1.0
                    add(e, lim.u_max, zero_x, zero_r)
                    add(-e, lim.u_max, zero_x, zero_r)
        for k in lim.z_steps:
            m = self.M[k - 1]
            ox = self.O[k - 1]
            if lim.z_hi is not None:  # z <= z_hi  <=>  M u <= z_hi - O x
                add(m, lim.z_hi, -ox, zero_r)
            if lim.z_lo is not None:  # z >= z_lo  <=>  -M u <= -z_lo + O x
                add(-m, -lim.z_lo, ox, zero_r)
        for k in lim.band_steps:
            m = self.M[k - 1]
            ox = self.O[k - 1]
            ek = np.zeros(f)
            ek[k - 1] = 1.0
            if lim.band is not None:
                add(m, lim.band, -ox, ek)  # z - r <= band  <=>  M u <= band + r - O x
                add(-m, lim.band, ox, -ek)  # r - z <= band  <=>  -M u <= band - r + O x
        if not rows:
            return np.zeros((0, n)), np.zeros(0), np.zeros((0, T))
        return np.array(rows), np.array(h0), np.array(Hth)
