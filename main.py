#!/usr/bin/env python3
"""
Full pipeline: download data → convert to CSV → EDA → train all models → test all models.
Run from repo root: python main.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import after repo root is established so path is correct
from modeling.utils.custom_logger import log_handler
from configuration.config_loader import config


#------------------------------------------------------
# Config
#------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent

# Scripts to run (paths relative to repo root)
DATASET_FOLDER = REPO_ROOT / "datasets"
DOWNLOAD_SCRIPT = REPO_ROOT / DATASET_FOLDER / "data_kaggle_download.py"
CONVERTER_SCRIPT = REPO_ROOT / DATASET_FOLDER / "aud_csv_converter_trad_ml.py"

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

#------------------------------------------------------
# Paths used to decide if a step is already done
#------------------------------------------------------
AUDIOS_DIR = REPO_ROOT / "datasets" / "audios"
CONVERTED_CSV_DIR = REPO_ROOT / "datasets" / "trad_ml_csv"

EDA_PLOTS_DIR = REPO_ROOT / "modeling" / "EDA" / "plots"
EDA_REPORT_DIR = REPO_ROOT / "modeling" / "EDA" / "eda_report"

TRAINED_TINY_CNN_MULTICLASS = REPO_ROOT / "_cnn_trained_models" / "multiclass"  / "tiny_cnn_audio_model"
TRAINED_ROBUST_CNN = REPO_ROOT / "_cnn_trained_models" / "multiclass"  / "robust_cnn_audio_model"
TRAINED_TINY_CNN_BINARY = REPO_ROOT / "_cnn_trained_models" / "binary" / "tiny_cnn_binary_model"
TRAINED_ML_MODELS_MULTICLASS = REPO_ROOT / "_ml_trained_models" / "multiclass"
TRAINED_ML_MODELS_BINARY = REPO_ROOT / "_ml_trained_models" / "binary"

LOGS_DIR = REPO_ROOT / "_logs"
TEST_LOG_FILES = [
    "testing_ml_models.log",
    "testing_cnn_models.log",
    "ml_models_training.log",
    "cnn_models_training.log",
]

#------------------------------------------------------
# Methods
#------------------------------------------------------
def _has_files(dir_path: Path, suffix: str) -> bool:
    """True if dir exists and contains at least one file with the given suffix (e.g. .wav)."""
    if not dir_path.is_dir():
        return False
    return any(f.suffix.lower() == suffix.lower() for f in dir_path.iterdir() if f.is_file())

def is_download_done() -> bool:
    """True if datasets/audios/ exists and has at least one .wav file."""
    return _has_files(AUDIOS_DIR, ".wav")

def is_conversion_done() -> bool:
    """True if datasets/converted_csv/ exists and has at least one .csv file."""
    return _has_files(CONVERTED_CSV_DIR, ".csv")

def is_eda_done() -> bool:
    """True if EDA plots and report outputs exist."""
    if not EDA_PLOTS_DIR.is_dir() or not _has_files(EDA_PLOTS_DIR, ".png"):
        return False
    report_md = EDA_REPORT_DIR / "07_eda_report.md"
    return report_md.is_file()

def is_training_done() -> bool:
    """True if all five model outputs exist (tiny_cnn, robust_cnn, ml_models, binary/tiny_cnn, binary/ml_models)."""
    def has_model(d: Path) -> bool:
        if not d.is_dir():
            return False
        return any(f.suffix.lower() in (".pkl", ".h5", ".keras") for f in d.iterdir() if f.is_file())
    return (
        has_model(TRAINED_ROBUST_CNN)
        and has_model(TRAINED_TINY_CNN_BINARY)
        and has_model(TRAINED_TINY_CNN_MULTICLASS)
        and has_model(TRAINED_ML_MODELS_BINARY)
        and has_model(TRAINED_ML_MODELS_MULTICLASS)
    )

def is_testing_done() -> bool:
    """True if all five test log files exist in logs/."""
    if not LOGS_DIR.is_dir():
        return False
    return all((LOGS_DIR / name).is_file() for name in TEST_LOG_FILES)

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
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        env=env,
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
    if is_download_done():
        log_handler.info("Step 1 skipped: data already in datasets/audios/")
        print("[SKIP] Step 1: data already downloaded (audios/ has .wav files)")
    else:
        if not run_sequential(DOWNLOAD_SCRIPT, "data_kaggle_download.py"):
            log_handler.error("Pipeline aborted: Step 1 (Download) failed")
            sys.exit(1)
        log_handler.info("Step 1 completed: Download dataset")

    # 2. Convert audio to CSV
    log_handler.info("Step 2: Convert audio to CSV")
    print("\n--- Step 2: Convert audio to CSV ---")
    if is_conversion_done():
        log_handler.info("Step 2 skipped: converted_csv/ already has .csv files")
        print("[SKIP] Step 2: conversion already done (converted_csv/ has .csv files)")
    else:
        if not run_sequential(CONVERTER_SCRIPT, "audio_to_csv_converter.py"):
            log_handler.error("Pipeline aborted: Step 2 (Convert) failed")
            sys.exit(1)
        log_handler.info("Step 2 completed: Convert audio to CSV")

    # 3. EDA (concurrent)
    log_handler.info("Step 3: EDA (concurrent)")
    print("\n--- Step 3: EDA (concurrent) ---")
    if is_eda_done():
        log_handler.info("Step 3 skipped: EDA outputs already exist (plots/ and eda_report/)")
        print("[SKIP] Step 3: EDA already done (plots/ and eda_report/ present)")
    else:
        if not run_concurrent(EDA_SCRIPTS, "EDA"):
            log_handler.error("Pipeline aborted: Step 3 (EDA) failed")
            sys.exit(1)
        log_handler.info("Step 3 completed: EDA")

    # 4. Train all models (concurrent)
    log_handler.info("Step 4: Train all models (concurrent)")
    print("\n--- Step 4: Train all models (concurrent) ---")
    if is_training_done():
        log_handler.info("Step 4 skipped: all model outputs already in trained_models/")
        print("[SKIP] Step 4: models already trained (trained_models/ has all outputs)")
    else:
        if not run_concurrent(TRAINING_SCRIPTS, "Training"):
            log_handler.error("Pipeline aborted: Step 4 (Training) failed")
            sys.exit(1)
        log_handler.info("Step 4 completed: Train all models")

    # 5. Test all models (concurrent; only after training is done)
    log_handler.info("Step 5: Test all models (concurrent)")
    print("\n--- Step 5: Test all models (concurrent) ---")
    if is_testing_done():
        log_handler.info("Step 5 skipped: all test log files already in logs/")
        print("[SKIP] Step 5: testing already done (logs/ has all test .log files)")
    else:
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
