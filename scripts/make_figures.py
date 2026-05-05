"""One-command figure generation for the non-training reports."""
from __future__ import annotations

import subprocess
import sys

COMMANDS = [
    [sys.executable, "scripts/eda_report.py"],
    [sys.executable, "scripts/feature_report.py"],
    [sys.executable, "scripts/baseline_report.py"],
]

for cmd in COMMANDS:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)
