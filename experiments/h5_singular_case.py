"""H5: Singular Hessian: initialisation dependence.

STATUS: STUB. Not implemented; exits with a message.

Test:  On the dual-drive plant with W3 = 0, warm-started solvers return null-space components that depend on initialisation, while the exact layer returns the minimum-norm one.
Kill criterion: If solvers agree, or the difference is immaterial in closed loop, C4 becomes a theoretical remark.
Needs: exact.singular (gate G2), models.dual_drive, baselines.qp_solvers (warm start)
Thresholds are fixed in docs/hypotheses.md before any run.
"""

import sys


def main() -> None:
    sys.exit("H5 is not implemented yet: see the docstring and docs/hypotheses.md")


if __name__ == "__main__":
    main()
