"""H3: The planner beats strong scheduling.

STATUS: STUB. Not implemented; exits with a message.

Test:  Cycle time at equal contour error against jerk-limited scheduling and convex time-optimal path tracking, on the standard test contours, at least three seeds with bootstrap intervals.
Kill criterion: If within noise, the planner claim is dropped and the thesis rests on the analytical part (paper P1).
Needs: baselines.topp, planner.*, eval (contour error, cycle time: to be added), contours (to be fixed with K. Tsabedze)
Thresholds are fixed in docs/hypotheses.md before any run.
"""

import sys


def main() -> None:
    sys.exit("H3 is not implemented yet: see the docstring and docs/hypotheses.md")


if __name__ == "__main__":
    main()
