"""H7: Capability maps match finite-difference re-solves.

STATUS: STUB. Not implemented; exits with a message.

Test:  Along a standard contour, the price and multipliers must match re-solves with perturbed limits to 1e-6 and identify the binding limit.
Kill criterion: If they disagree, the map is not reported.
Needs: exact.price (value_gradient, margin_price), eval, contours
Thresholds are fixed in docs/hypotheses.md before any run.
"""

import sys


def main() -> None:
    sys.exit("H7 is not implemented yet: see the docstring and docs/hypotheses.md")


if __name__ == "__main__":
    main()
