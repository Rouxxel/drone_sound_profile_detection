#!/usr/bin/env python3
"""
Full pipeline: download data → convert to CSV → EDA → train all models → test all models.
Run from repo root: python main.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parent

# Scripts to run (paths relative to repo root)
DOWNLOAD_SCRIPT = REPO_ROOT / "datasets" / "data_kaggle_download.py"
CONVERTER_SCRIPT = REPO_ROOT / "datasets" / "audio_to_csv_converter.py"

EDA_SCRIPTS = [
    REPO_ROOT / "modeling" / "EDA" / "plot_converted_csv.py",
    REPO_ROOT / "modeling" / "EDA" / "eda_summary_report.py",
]

TRAINING_SCRIPTS = [
    REPO_ROOT / "modeling" / "models" / "multiclass" / "tiny_cnn_model.py",
    REPO_ROOT / "modeling" / "models" / "multiclass" / "ml_models.py",
    REPO_ROOT / "modeling" / "models" / "multiclass" / "robust_cnn_model.py",
    REPO_ROOT / "modeling" / "models" / "binary" / "tiny_cnn_binary.py",
    REPO_ROOT / "modeling" / "models" / "binary" / "ml_models_binary.py",
]

TESTING_SCRIPTS = [
    REPO_ROOT / "modeling" / "models" / "multiclass" / "testing" / "test_tiny_model.py",
    REPO_ROOT / "modeling" / "models" / "multiclass" / "testing" / "test_ml_models.py",
    REPO_ROOT / "modeling" / "models" / "multiclass" / "testing" / "test_robust_model.py",
    REPO_ROOT / "modeling" / "models" / "binary" / "testing" / "test_tiny_cnn_binary.py",
    REPO_ROOT / "modeling" / "models" / "binary" / "testing" / "test_ml_models_binary.py",
]


def run_script(script_path: Path, label: str = None) -> tuple[str, int]:
    """Run a Python script from repo root; return (label, returncode)."""
    name = label or script_path.name
    if not script_path.exists():
        print(f"[SKIP] {name}: script not found at {script_path}")
        return (name, -1)
    print(f"[RUN] {name} ...")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        capture_output=False,
    )
    return (name, result.returncode)


def run_sequential(script_path: Path, label: str = None) -> bool:
    """Run one script; return True if success."""
    name, code = run_script(script_path, label)
    if code != 0:
        print(f"[FAIL] {name} exited with code {code}")
        return False
    print(f"[OK] {name}")
    return True


def run_concurrent(script_paths: list[Path], step_name: str) -> bool:
    """Run all scripts concurrently; return True only if all succeed."""
    with ThreadPoolExecutor(max_workers=len(script_paths)) as executor:
        futures = {
            executor.submit(run_script, p, p.name): p
            for p in script_paths
        }
        results = []
        for future in as_completed(futures):
            name, code = future.result()
            results.append((name, code))
            if code != 0:
                print(f"[FAIL] {name} exited with code {code}")
            else:
                print(f"[OK] {name}")

    failed = [n for n, c in results if c != 0]
    if failed:
        print(f"[STEP FAILED] {step_name}: {len(failed)} script(s) failed: {failed}")
        return False
    print(f"[STEP OK] {step_name}: all {len(script_paths)} script(s) completed.")
    return True


def main():
    print("=" * 60)
    print("Pipeline: download → convert → EDA → train → test")
    print("=" * 60)

    # 1. Download data
    print("\n--- Step 1: Download dataset ---")
    if not run_sequential(DOWNLOAD_SCRIPT, "data_kaggle_download.py"):
        sys.exit(1)

    # 2. Convert audio to CSV
    print("\n--- Step 2: Convert audio to CSV ---")
    if not run_sequential(CONVERTER_SCRIPT, "audio_to_csv_converter.py"):
        sys.exit(1)

    # 3. EDA (concurrent)
    print("\n--- Step 3: EDA (concurrent) ---")
    if not run_concurrent(EDA_SCRIPTS, "EDA"):
        sys.exit(1)

    # 4. Train all models (concurrent)
    print("\n--- Step 4: Train all models (concurrent) ---")
    if not run_concurrent(TRAINING_SCRIPTS, "Training"):
        sys.exit(1)

    # 5. Test all models (concurrent; only after training is done)
    print("\n--- Step 5: Test all models (concurrent) ---")
    if not run_concurrent(TESTING_SCRIPTS, "Testing"):
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Pipeline finished successfully.")
    print("Testing results (accuracy, confusion matrix, etc.) are in logs/:")
    print("  - tiny_cnn_testing.log, ml_models_testing.log, robust_cnn_testing.log")
    print("  - tiny_cnn_binary_testing.log, ml_models_binary_testing.log")
    print("=" * 60)


if __name__ == "__main__":
    main()
