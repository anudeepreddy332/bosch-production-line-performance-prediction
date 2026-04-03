from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run_step(name: str, cmd: list[str]) -> bool:
    print(f"[START] {name}")
    try:
        result = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())
        print(f"[OK] {name}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[FAIL] {name}")
        if exc.stdout:
            print(exc.stdout)
        if exc.stderr:
            print(exc.stderr)
        return False


if __name__ == "__main__":
    steps = [
        ("Build Decision System", [sys.executable, "scripts/build_decision_summary.py"]),
        ("Run Batch Simulation", [sys.executable, "scripts/run_batch_simulation.py"]),
        ("Run Drift Monitoring", [sys.executable, "scripts/run_drift_monitoring.py"]),
    ]

    failed = []
    for name, cmd in steps:
        if not run_step(name, cmd):
            failed.append(name)

    print("=" * 72)
    if failed:
        print("System run completed with failures:")
        for name in failed:
            print(f" - {name}")
        sys.exit(1)

    print("System run completed successfully.")
    print("Generated outputs:")
    print(" - outputs/production_decision_summary.json")
    print(" - outputs/batch_simulation_summary.json")
    print(" - outputs/monitoring/evidently_summary.json")
