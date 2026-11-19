"""H2: Training through the exact layer is at least as good as an OptNet-style layer.

STATUS: STUB. Not implemented; exits with a message.

Test:  Same planner, trained (a) through the exact layer, (b) through an OptNet-style QP layer at tolerance 1e-3, (c) at 1e-6. Compare final loss, gradient error against finite differences, and wall-clock.
Kill criterion: If (c) matches the exact layer in loss and time, contribution C2 reduces to determinism plus the certificate.
Needs: planner.layer.ExactMPCLayer, baselines.optnet_layer.OptNetLayer, planner.network.PlannerNet, planner.loss
Thresholds are fixed in docs/hypotheses.md before any run.
"""

import sys


def main() -> None:
    sys.exit("H2 is not implemented yet: see the docstring and docs/hypotheses.md")


if __name__ == "__main__":
    main()
