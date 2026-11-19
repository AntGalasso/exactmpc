"""H4: Safety by construction.

STATUS: STUB. Not implemented; exits with a message.

Test:  Zero limit violations over randomised references inside the guard set, while the naive network of Hose et al. violates limits.
Kill criterion: Any violation inside the guard set: T2 is corrected or restricted.
Needs: interface.GuardSet (T2), baselines.hose_verified, planner.*
Thresholds are fixed in docs/hypotheses.md before any run.
"""

import sys


def main() -> None:
    sys.exit("H4 is not implemented yet: see the docstring and docs/hypotheses.md")


if __name__ == "__main__":
    main()
