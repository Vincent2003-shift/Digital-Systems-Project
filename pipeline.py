"""
pipeline.py — Run the full cheat-detection pipeline in order.

Steps:
  1. features.py  — extract behavioural features from .npy files
  2. train.py     — train Isolation Forest, Random Forest, XGBoost
  3. evaluate.py  — metrics, ROC curves, confusion matrices
  4. visualise.py — dissertation-quality exploratory plots
"""
import subprocess
import sys
from pathlib import Path

STEPS = [
    "features.py",
    "train.py",
    "evaluate.py",
    "visualise.py",
]

ROOT_DIR = Path(__file__).resolve().parent


def run_step(script):
    print(f"\n{'#'*60}")
    print(f"#  {script}")
    print(f"{'#'*60}\n")
    result = subprocess.run(
        [sys.executable, str(ROOT_DIR / script)],
        cwd=ROOT_DIR,
        check=False,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed with exit code {result.returncode}")
        print("Stopping pipeline.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    print("=" * 60)
    print("  Cheat Detection Pipeline")
    print("  Behavioral Anomaly Detection in Online Multiplayer Games")
    print("=" * 60)

    for step in STEPS:
        run_step(step)

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print(f"  Results  -> {ROOT_DIR / 'results'}")
    print(f"  Figures  -> {ROOT_DIR / 'figures'}")
    print(f"  Models   -> {ROOT_DIR / 'models'}")
    print("=" * 60)
