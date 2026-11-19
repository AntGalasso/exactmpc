"""Differentiable exact layer (torch autograd Function).

STATUS: STUB.

forward(theta):  solve_exact for a batch of parameters theta = [x; r] with ExactQP, return the
                 applied input u_0 and the price Delta; save u, nu for backward.
backward:        use the closed forms, no solver in the backward pass and no tolerance:
                   dDelta/dtheta = -(G P^{-1} Qth + Hth)' nu         (exactmpc.exact.price)
                   du/dtheta     = -P^{-1}(Qth + G_J' dnu/dtheta), with dnu_J/dtheta = -S_J^{-1} (G_J P^{-1} Qth + Hth_J)
                 (derive and test du/dtheta against finite differences before using it).
Test before use: gradient check against finite differences, and against an OptNet-style layer
at tolerance 1e-3 / 1e-6 (hypothesis H2).
"""


class ExactMPCLayer:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("see module docstring")
