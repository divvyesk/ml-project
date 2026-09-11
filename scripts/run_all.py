"""Run the whole project, Week 5 through Week 15, in order.

Each week is a separate process so a failure points at one week rather than
at a single long traceback. Weeks depend on the artifacts of earlier weeks,
so the order here is the order they must run in.
"""

import _bootstrap  # noqa: F401

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

WEEKS = [
    ("Week 5  - Dataset understanding", "week05_understanding.py"),
    ("Week 6  - Cleaning and preprocessing", "week06_cleaning.py"),
    ("Week 7  - Exploratory data analysis", "week07_eda.py"),
    ("Week 8  - Feature engineering", "week08_features.py"),
    ("Week 9  - Baseline models", "week09_baseline.py"),
    ("Week 10 - Model training", "week10_train.py"),
    ("Week 11 - Performance evaluation", "week11_evaluate.py"),
    ("Week 12 - Tuning and improvement", "week12_tune.py"),
    ("Week 13 - Prototype check", "week13_prototype_check.py"),
    ("Week 14 - Final report", "week14_final_report.py"),
    ("Week 15 - Presentation pack", "week15_presentation.py"),
]


def main():
    started = time.time()

    for label, script in WEEKS:
        print()
        print("=" * 72)
        print(label)
        print("=" * 72)
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / script)], cwd=str(ROOT)
        )
        if result.returncode != 0:
            print()
            print("FAILED at {0} ({1}).".format(label, script))
            return result.returncode

    elapsed = time.time() - started
    print()
    print("=" * 72)
    print("Weeks 5-15 complete in {0:.0f}s.".format(elapsed))
    print("=" * 72)
    print("Final report:        reports/final_report.md")
    print("Model card:          docs/MODEL_CARD.md")
    print("Presentation:        reports/presentation_outline.md")
    print("Viva questions:      docs/VIVA_QA.md")
    print("Week 13 prototype:   streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
