#!/usr/bin/env python3
"""
Full pipeline: download data → convert to CSV → EDA → train all models → test all models.
Run from repo root: python main.py
Logs every step to modeling_training.log in the repo root.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import after repo root is established so path is correct
from modeling.utils.custom_logger import log_handler

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
        msg = f"[SKIP] {name}: script not found at {script_path}"
        log_handler.warning(msg)
        print(msg)
        return (name, -1)
    log_handler.info(f"Running script: {name}")
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
        log_handler.error(f"Script failed: {name} (exit code {code})")
        print(f"[FAIL] {name} exited with code {code}")
        return False
    log_handler.info(f"Script completed successfully: {name}")
    print(f"[OK] {name}")
    return True


def run_concurrent(script_paths: list[Path], step_name: str) -> bool:
    """Run all scripts concurrently; return True only if all succeed."""
    log_handler.info(f"Step '{step_name}': running {len(script_paths)} script(s) concurrently")
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
                log_handler.error(f"Script failed: {name} (exit code {code})")
                print(f"[FAIL] {name} exited with code {code}")
            else:
                log_handler.info(f"Script completed: {name}")
                print(f"[OK] {name}")

    failed = [n for n, c in results if c != 0]
    if failed:
        log_handler.error(f"Step '{step_name}' failed: {len(failed)} script(s) failed: {failed}")
        print(f"[STEP FAILED] {step_name}: {len(failed)} script(s) failed: {failed}")
        return False
    log_handler.info(f"Step '{step_name}' completed: all {len(script_paths)} script(s) succeeded")
    print(f"[STEP OK] {step_name}: all {len(script_paths)} script(s) completed.")
    return True


def main():
    log_handler.info("Pipeline started: download → convert → EDA → train → test")
    log_handler.info(f"Repo root: {REPO_ROOT}")
    print("=" * 60)
    print("Pipeline: download → convert → EDA → train → test")
    print("=" * 60)

    # 1. Download data
    log_handler.info("Step 1: Download dataset")
    print("\n--- Step 1: Download dataset ---")
    if not run_sequential(DOWNLOAD_SCRIPT, "data_kaggle_download.py"):
        log_handler.error("Pipeline aborted: Step 1 (Download) failed")
        sys.exit(1)
    log_handler.info("Step 1 completed: Download dataset")

    # 2. Convert audio to CSV
    log_handler.info("Step 2: Convert audio to CSV")
    print("\n--- Step 2: Convert audio to CSV ---")
    if not run_sequential(CONVERTER_SCRIPT, "audio_to_csv_converter.py"):
        log_handler.error("Pipeline aborted: Step 2 (Convert) failed")
        sys.exit(1)
    log_handler.info("Step 2 completed: Convert audio to CSV")

    # 3. EDA (concurrent)
    log_handler.info("Step 3: EDA (concurrent)")
    print("\n--- Step 3: EDA (concurrent) ---")
    if not run_concurrent(EDA_SCRIPTS, "EDA"):
        log_handler.error("Pipeline aborted: Step 3 (EDA) failed")
        sys.exit(1)
    log_handler.info("Step 3 completed: EDA")

    # 4. Train all models (concurrent)
    log_handler.info("Step 4: Train all models (concurrent)")
    print("\n--- Step 4: Train all models (concurrent) ---")
    if not run_concurrent(TRAINING_SCRIPTS, "Training"):
        log_handler.error("Pipeline aborted: Step 4 (Training) failed")
        sys.exit(1)
    log_handler.info("Step 4 completed: Train all models")

    # 5. Test all models (concurrent; only after training is done)
    log_handler.info("Step 5: Test all models (concurrent)")
    print("\n--- Step 5: Test all models (concurrent) ---")
    if not run_concurrent(TESTING_SCRIPTS, "Testing"):
        log_handler.error("Pipeline aborted: Step 5 (Testing) failed")
        sys.exit(1)
    log_handler.info("Step 5 completed: Test all models")

    log_handler.info("Pipeline finished successfully")
    log_handler.info("Testing results are in logs/: tiny_cnn_testing.log, ml_models_testing.log, robust_cnn_testing.log, tiny_cnn_binary_testing.log, ml_models_binary_testing.log")
    print("\n" + "=" * 60)
    print("Pipeline finished successfully.")
    print("Testing results (accuracy, confusion matrix, etc.) are in logs/:")
    print("  - tiny_cnn_testing.log, ml_models_testing.log, robust_cnn_testing.log")
    print("  - tiny_cnn_binary_testing.log, ml_models_binary_testing.log")
    print("=" * 60)


if __name__ == "__main__":
    main()
