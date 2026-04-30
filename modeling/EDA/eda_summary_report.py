"""
EDA Summary Report
-----------------
Goes through each CSV in datasets/converted_csv/, computes dataset-level stats,
and saves plots + a markdown report into modeling/EDA/eda_report/.
Run from any directory; paths are relative to this script.
"""

import os
import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#------------------------------------------------------
# Paths
#------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent #EDA/
REPO_ROOT = SCRIPT_DIR.parent.parent #root/
AUDIOS_FOLDER = REPO_ROOT / "datasets" / "audios"

#------------------------------------------------------
# Config
#------------------------------------------------------
sys.path.append(str(REPO_ROOT))
from configuration.config_loader import config
SAMP_RATE = config["eda"]["summary_report"]["samp_rate"]
HOP_LENGTH = config["eda"]["summary_report"]["hop_length"]
N_MFCC = config["eda"]["summary_report"]["n_mfcc"]
DURATION_SHORT_S = config["eda"]["summary_report"]["duration_short.s"]
DURATION_LONG_S = config["eda"]["summary_report"]["duration_long.s"]
MEAN_RMS_SILENCE_THRESHOLD = config["eda"]["summary_report"]["mean_rms_silence_threshold"]
SPECIFIC_DIR = config["eda"]["summary_report"]["output_folder_str"]
CSV_FOLDER = REPO_ROOT / "datasets" / config["audio_converters"]["trad_ml_models"]["output_folder_str"]
OUTPUT_DIR = SCRIPT_DIR / SPECIFIC_DIR

#------------------------------------------------------
# Methods
#------------------------------------------------------
def extract_statistical_features(mfcc: np.ndarray) -> np.ndarray:
    """Same as in ml_models: mean, std, min, max, median, p25, p75, delta mean/std."""
    mfcc = np.asarray(mfcc, dtype=np.float32)
    features = []
    features.extend(np.mean(mfcc, axis=0))
    features.extend(np.std(mfcc, axis=0))
    features.extend(np.min(mfcc, axis=0))
    features.extend(np.max(mfcc, axis=0))
    features.extend(np.median(mfcc, axis=0))
    features.extend(np.percentile(mfcc, 25, axis=0))
    features.extend(np.percentile(mfcc, 75, axis=0))
    delta = np.diff(mfcc, axis=0)
    if len(delta) > 0:
        features.extend(np.mean(delta, axis=0))
        features.extend(np.std(delta, axis=0))
    else:
        features.extend(np.zeros(mfcc.shape[1]))
        features.extend(np.zeros(mfcc.shape[1]))
    return np.array(features, dtype=np.float32)

def load_all_csv_summaries():
    """Load each CSV, compute per-file stats and optional RMS; return list of dicts."""
    if not CSV_FOLDER.exists():
        raise FileNotFoundError(f"CSV folder not found: {CSV_FOLDER}")

    csv_files = sorted(f for f in os.listdir(CSV_FOLDER) if f.lower().endswith(".csv"))
    rows = []

    for csv_file in csv_files:
        csv_path = CSV_FOLDER / csv_file
        base_name = csv_path.stem
        class_name = base_name.split("_")[0]

        mfcc_df = pd.read_csv(csv_path)
        mfcc = mfcc_df.values.astype(np.float32)
        n_frames, n_mfcc = mfcc.shape
        duration_sec = n_frames * HOP_LENGTH / SAMP_RATE

        feat_vec = extract_statistical_features(mfcc)
        mean_mfcc = np.mean(mfcc, axis=0)
        std_mfcc = np.std(mfcc, axis=0)

        row = {
            "file": csv_file,
            "class": class_name,
            "n_frames": n_frames,
            "duration_sec": duration_sec,
            "feat_vec": feat_vec,
            "mean_mfcc": mean_mfcc,
            "std_mfcc": std_mfcc,
        }

        audio_path = AUDIOS_FOLDER / f"{base_name}.wav"
        if audio_path.exists():
            try:
                y, _ = librosa.load(str(audio_path), sr=SAMP_RATE)
                rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
                row["mean_rms"] = float(np.mean(rms))
                row["max_rms"] = float(np.max(rms))
            except Exception:
                row["mean_rms"] = np.nan
                row["max_rms"] = np.nan
        else:
            row["mean_rms"] = np.nan
            row["max_rms"] = np.nan

        rows.append(row)

    return rows

def plot_class_distribution(df: pd.DataFrame, out_dir: Path):
    """Bar chart: count per class."""
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = df["class"].value_counts().sort_index()
    counts.plot(kind="bar", ax=ax, color=["#2ecc71", "#3498db", "#95a5a6"], edgecolor="black")
    ax.set_title("Class distribution (number of files)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "01_class_distribution.png", dpi=150)
    plt.close()
    print("  Saved 01_class_distribution.png")

def plot_duration_distribution(df: pd.DataFrame, out_dir: Path):
    """Boxplot of duration per class."""
    fig, ax = plt.subplots(figsize=(8, 4))
    classes = df["class"].unique()
    data = [df.loc[df["class"] == c, "duration_sec"].values for c in sorted(classes)]
    bp = ax.boxplot(data, tick_labels=sorted(classes), patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(["#2ecc71", "#3498db", "#95a5a6"][i % 3])
    ax.set_title("Duration distribution per class (seconds)")
    ax.set_ylabel("Duration (s)")
    plt.tight_layout()
    plt.savefig(out_dir / "02_duration_by_class.png", dpi=150)
    plt.close()
    print("  Saved 02_duration_by_class.png")

def plot_mfcc_mean_by_class(rows: list, out_dir: Path):
    """Mean MFCC profile per class (line plot) + per-coefficient boxplots by class."""
    df = pd.DataFrame(rows)
    classes = sorted(df["class"].unique())
    n_features = len(df["mean_mfcc"].iloc[0])  # Works for 14, 21, or more

    # -------------------------------
    # 03: Line plot of mean profile
    # -------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2ecc71", "#3498db", "#95a5a6"]
    for i, c in enumerate(classes):
        means = np.array([r["mean_mfcc"] for r in rows if r["class"] == c])
        profile = np.mean(means, axis=0)
        ax.plot(range(n_features), profile, "o-", label=c, color=colors[i % 3], linewidth=2)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Mean value (across files)")
    ax.set_title("Mean feature profile per class")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "03_mean_by_class.png", dpi=150)
    plt.close()
    print("  Saved 03_mean_by_class.png")

    # -------------------------------
    # 04: Boxplots per coefficient
    # -------------------------------
    # Determine subplot grid automatically
    n_cols = 7
    n_rows = math.ceil(n_features / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2.5, n_rows*2.5))
    axes = axes.flatten()

    for coef in range(n_features):
        ax = axes[coef]
        data_by_class = [[r["mean_mfcc"][coef] for r in rows if r["class"] == c] for c in classes]
        bp = ax.boxplot(data_by_class, tick_labels=classes, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(["#2ecc71", "#3498db", "#95a5a6"][i % 3])
        ax.set_title(f"Feature {coef}")
        ax.set_xticklabels(classes, rotation=45, ha="right")

    # Hide any unused axes
    for i in range(n_features, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Per-feature boxplots by class", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "04_features_by_class.png", dpi=150)
    plt.close()
    print("  Saved 04_features_by_class.png")

def save_summary_stats_table(rows: list, out_dir: Path):
    """Per-class summary stats (mean of mean_mfcc, etc.) as CSV."""
    df = pd.DataFrame(rows)
    classes = sorted(df["class"].unique())
    n_mfcc = len(df["mean_mfcc"].iloc[0])

    records = []
    for c in classes:
        sub = df[df["class"] == c]
        rec = {"class": c, "n_files": len(sub), "duration_sec_mean": sub["duration_sec"].mean(), "duration_sec_std": sub["duration_sec"].std()}
        for i in range(n_mfcc):
            means = [r["mean_mfcc"][i] for r in rows if r["class"] == c]
            rec[f"mean_mfcc_{i}_mean"] = np.mean(means)
            rec[f"mean_mfcc_{i}_std"] = np.std(means)
        if "mean_rms" in sub.columns and sub["mean_rms"].notna().any():
            rec["mean_rms_mean"] = sub["mean_rms"].mean()
            rec["mean_rms_std"] = sub["mean_rms"].std()
        records.append(rec)

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(out_dir / "05_summary_stats_by_class.csv", index=False)
    print("  Saved 05_summary_stats_by_class.csv")

def plot_pca_separability(rows: list, out_dir: Path):
    """PCA on full feature vectors; scatter colored by class."""
    X = np.array([r["feat_vec"] for r in rows])
    y = [r["class"] for r in rows]
    classes = sorted(set(y))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2ecc71", "#3498db", "#95a5a6"]
    for i, c in enumerate(classes):
        mask = [cls == c for cls in y]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=c, alpha=0.7, c=colors[i % 3], edgecolors="black", s=50)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA of per-file statistical features (separability by class)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "06_pca_by_class.png", dpi=150)
    plt.close()
    print("  Saved 06_pca_by_class.png")

def quality_checks(rows: list) -> dict:
    """Return short, long, and near-silence file lists."""
    short = [r["file"] for r in rows if r["duration_sec"] < DURATION_SHORT_S]
    long_ = [r["file"] for r in rows if r["duration_sec"] > DURATION_LONG_S]
    silent = []
    for r in rows:
        mr = r.get("mean_rms")
        if mr is not None and not np.isnan(mr) and mr < MEAN_RMS_SILENCE_THRESHOLD:
            silent.append(r["file"])
    return {"short": short, "long": long_, "silent": silent}

def write_report(rows: list, out_dir: Path, quality: dict):
    """Write markdown report with counts, duration stats, quality flags, figure refs."""
    df = pd.DataFrame(rows)
    n_total = len(rows)
    classes = sorted(df["class"].unique())

    lines = [
        "# EDA Summary Report",
        "",
        "## Dataset overview",
        f"- **Total files:** {n_total}",
        f"- **Classes:** {', '.join(classes)}",
        "",
        "### Class distribution",
        "",
    ]
    for c in classes:
        count = (df["class"] == c).sum()
        lines.append(f"- **{c}:** {count} files")
    lines.extend(["", "### Duration (seconds)", ""])
    lines.append(f"- **Mean:** {df['duration_sec'].mean():.2f}")
    lines.append(f"- **Std:** {df['duration_sec'].std():.2f}")
    lines.append(f"- **Min:** {df['duration_sec'].min():.2f}")
    lines.append(f"- **Max:** {df['duration_sec'].max():.2f}")
    lines.extend(["", "### Quality checks", ""])
    lines.append(f"- **Very short (< {DURATION_SHORT_S} s):** {len(quality['short'])} files")
    if quality["short"]:
        lines.append("  - " + ", ".join(quality["short"][:10]) + (" ..." if len(quality["short"]) > 10 else ""))
    lines.append(f"- **Very long (> {DURATION_LONG_S} s):** {len(quality['long'])} files")
    if quality["long"]:
        lines.append("  - " + ", ".join(quality["long"][:10]) + (" ..." if len(quality["long"]) > 10 else ""))
    lines.append(f"- **Near-silence (mean RMS < {MEAN_RMS_SILENCE_THRESHOLD}):** {len(quality['silent'])} files")
    if quality["silent"]:
        lines.append("  - " + ", ".join(quality["silent"][:10]) + (" ..." if len(quality["silent"]) > 10 else ""))
    lines.extend([
        "",
        "## Generated figures",
        "",
        "- `01_class_distribution.png` – Count per class",
        "- `02_duration_by_class.png` – Duration boxplot per class",
        "- `03_mfcc_mean_by_class.png` – Mean MFCC profile per class (line)",
        "- `04_mfcc_coef_by_class.png` – Per-coefficient boxplots by class",
        "- `05_summary_stats_by_class.csv` – Summary statistics table",
        "- `06_pca_by_class.png` – PCA of per-file features, colored by class",
        "",
    ])

    report_path = out_dir / "07_eda_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("  Saved 07_eda_report.md")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"CSV folder:  {CSV_FOLDER}")
    print(f"Output:     {OUTPUT_DIR}\n")

    print("Loading all CSV files...")
    rows = load_all_csv_summaries()
    if not rows:
        print("No CSV files found.")
        return
    print(f"  Loaded {len(rows)} files.\n")

    df = pd.DataFrame(rows)
    quality = quality_checks(rows)

    print("Generating plots and tables...")
    plot_class_distribution(df, OUTPUT_DIR)
    plot_duration_distribution(df, OUTPUT_DIR)
    plot_mfcc_mean_by_class(rows, OUTPUT_DIR)
    save_summary_stats_table(rows, OUTPUT_DIR)
    plot_pca_separability(rows, OUTPUT_DIR)
    write_report(rows, OUTPUT_DIR, quality)

    print("\nEDA summary report done. All outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
