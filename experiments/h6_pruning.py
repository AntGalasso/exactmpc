"""H6: Pruning visits far fewer than 2^kappa - 1 candidates.

STATUS: STUB. Not implemented; exits with a message.

Test:  Count candidates visited by exact.pruning.solve_exact_pruned over the state space on plants A to C, for kappa up to 6 and beyond.
Kill criterion: If more than half of all subsets occur, the pruning claim is withdrawn. Soundness is tested in tests/test_pruning.py; efficiency is NOT established.
Needs: exact.pruning (single QP; a batched version is needed)
Thresholds are fixed in docs/hypotheses.md before any run.
"""

import sys


def main() -> None:
    sys.exit("H6 is not implemented yet: see the docstring and docs/hypotheses.md")


if __name__ == "__main__":
    main()
