#!/usr/bin/env python3
"""
Analyze handcrafted prosodic features comparing high vs. low prosody score samples.

Uses pre-extracted features from:
  data/speechocean762/tr_handcrafted_feats.pkl  (train)
  data/speechocean762/te_handcrafted_feats.pkl  (test)

Each pkl is a dict:  { wav_rel_path -> torch.Tensor(T, 50) }
  cols  0- 7 : Loudness (8 A-weighted bands)
  col   8    : Pitch F0
  col   9    : Periodicity
  cols 10-49 : PPG (40 phonetic posteriors)

Usage:
  python -m prosody_scorer.analyze_handcrafted \
      --dataset_dir  data/speechocean762/so762 \
      --label_dir    data/speechocean762 \
      --output_dir   exp/handcrafted_analysis \
      --split        test \
      --n_samples    30 \
      --high_thresh  9 \
      --low_thresh   5
"""

import argparse
import csv
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCORE_LABELS = ["accuracy", "completeness", "fluency", "prosodic", "total"]
PROSODIC_IDX = 3
PPG_DIM = 40
LOUDNESS_DIM = 8
PERIODICITY_THRESHOLD = 0.5

PPG_PHONEMES = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    "SIL",
]

COLORS = {"high": "#E63946", "low": "#457B9D"}

# SIL run thresholds (in PPG frames; promonet hop ≈ 16 ms/frame)
SIL_IDX = 39
SIL_PROB_THRESHOLD = 0.5  # posterior threshold to call a frame "silent"
HESITATION_MIN_FRAMES = 10  # ≥ 160 ms  → hesitation / between-word pause
EPENTHESIS_MAX_FRAMES = 2  # ≤  32 ms  → epenthetic / within-word SIL burst


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_wav_scp(scp_path: Path):
    """Return list of (utt_id, relative_path) from a wav.scp file."""
    entries = []
    with open(scp_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                entries.append((parts[0], parts[1]))
    return entries


# ---------------------------------------------------------------------------
# Feature loading from pre-extracted pkl
# ---------------------------------------------------------------------------


def load_feats_from_pkl(feats: dict, rel_path: str):
    """Split a (T, 50) feature tensor into component arrays.

    Returns:
        loudness    (8, T) numpy array
        pitch       (1, T) numpy array
        periodicity (1, T) numpy array
        ppg         (40, T) numpy array
    or (None, None, None, None) if the key is missing.
    """
    if rel_path not in feats:
        return None, None, None, None
    t = feats[rel_path]
    arr = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)  # (T, 50)
    loudness = arr[:, :8].T  # (8,  T)
    pitch = arr[:, 8:9].T  # (1,  T)
    periodicity = arr[:, 9:10].T  # (1,  T)
    ppg = arr[:, 10:].T  # (40, T)
    return loudness, pitch, periodicity, ppg


# ---------------------------------------------------------------------------
# Per-utterance statistics
# ---------------------------------------------------------------------------


def compute_stats(loudness, pitch, periodicity, ppg):
    """Compute scalar/vector statistics per utterance.

    Returns a dict with:
        loudness_mean   (8,)
        loudness_std    (8,)
        pitch_mean      scalar  (voiced frames only)
        pitch_std       scalar
        pitch_range     scalar
        pitch_contour   (T,)
        voiced_ratio    scalar
        period_mean     scalar
        period_std      scalar
        period_contour  (T,)
        ppg_entropy_mean scalar
        ppg_entropy_std  scalar
        ppg_mean_dist   (40,)
        feature_vec     (1D feature vector for t-SNE)
    """
    d = {}

    # --- Loudness ---------------------------------------------------------
    if loudness is not None:
        # loudness shape: (8, T) or (T,)
        loud = np.atleast_2d(loudness)  # (8, T) or (1, T)
        d["loudness_mean"] = loud.mean(axis=-1)  # (bands,)
        d["loudness_std"] = loud.std(axis=-1)
        d["loudness_contour"] = loud.mean(axis=0)  # (T,)  mean across bands
    else:
        d["loudness_mean"] = np.zeros(LOUDNESS_DIM)
        d["loudness_std"] = np.zeros(LOUDNESS_DIM)
        d["loudness_contour"] = np.array([])

    # --- Pitch / Periodicity ----------------------------------------------
    if pitch is not None and periodicity is not None:
        p = pitch.squeeze()  # (T,)
        v = periodicity.squeeze()  # (T,)
        voiced = v > PERIODICITY_THRESHOLD
        d["voiced_ratio"] = voiced.mean()
        d["period_mean"] = v.mean()
        d["period_std"] = v.std()
        d["period_contour"] = v

        if voiced.any():
            voiced_p = p[voiced]
            d["pitch_mean"] = float(voiced_p.mean())
            d["pitch_std"] = float(voiced_p.std())
            d["pitch_range"] = float(voiced_p.max() - voiced_p.min())
        else:
            d["pitch_mean"] = 0.0
            d["pitch_std"] = 0.0
            d["pitch_range"] = 0.0

        # Full pitch contour with 0 for unvoiced
        d["pitch_contour"] = np.where(voiced, p, 0.0)
    else:
        d.update(
            voiced_ratio=0.0,
            period_mean=0.0,
            period_std=0.0,
            pitch_mean=0.0,
            pitch_std=0.0,
            pitch_range=0.0,
            pitch_contour=np.array([]),
            period_contour=np.array([]),
        )

    # --- PPG --------------------------------------------------------------
    pg = None
    if ppg is not None:
        pg = np.atleast_2d(ppg)  # (40, T)
        pg = np.clip(pg, 1e-10, 1.0)
        entropy = -(pg * np.log(pg)).sum(axis=0)  # (T,)
        d["ppg_entropy_mean"] = float(entropy.mean())
        d["ppg_entropy_std"] = float(entropy.std())
        d["ppg_mean_dist"] = pg.mean(axis=-1)  # (40,)
    else:
        d["ppg_entropy_mean"] = 0.0
        d["ppg_entropy_std"] = 0.0
        d["ppg_mean_dist"] = np.zeros(PPG_DIM)

    # --- SIL run analysis (hesitation vs epenthesis) ----------------------
    if pg is not None:
        sil_col = pg[SIL_IDX]  # (T,) already clipped
        sil_mask = sil_col > SIL_PROB_THRESHOLD

        # RLE: list of (is_sil, run_length)
        runs = []
        cur_val, cur_len = bool(sil_mask[0]), 1
        for b in sil_mask[1:]:
            if bool(b) == cur_val:
                cur_len += 1
            else:
                runs.append((cur_val, cur_len))
                cur_val, cur_len = bool(b), 1
        runs.append((cur_val, cur_len))

        # Collect only interior SIL runs (exclude leading/trailing silence)
        interior_sil = []
        for i, (is_sil, length) in enumerate(runs):
            if not is_sil:
                continue
            if i == 0 or i == len(runs) - 1:
                continue  # skip edge silence (recording padding)
            interior_sil.append(length)

        d["sil_rate"] = float(sil_mask.mean())
        d["sil_run_lengths"] = interior_sil
        d["total_sil_runs"] = len(interior_sil)
        d["hesitation_count"] = sum(
            1 for l in interior_sil if l >= HESITATION_MIN_FRAMES
        )
        d["epenthesis_count"] = sum(
            1 for l in interior_sil if l <= EPENTHESIS_MAX_FRAMES
        )
        d["mean_sil_run_len"] = float(np.mean(interior_sil)) if interior_sil else 0.0
    else:
        d["sil_rate"] = 0.0
        d["sil_run_lengths"] = []
        d["total_sil_runs"] = 0
        d["hesitation_count"] = 0
        d["epenthesis_count"] = 0
        d["mean_sil_run_len"] = 0.0

    # --- Flat feature vector for t-SNE -----------------------------------
    vec = np.concatenate(
        [
            d["loudness_mean"],
            d["loudness_std"],
            [d["pitch_mean"], d["pitch_std"], d["pitch_range"]],
            [d["voiced_ratio"], d["period_mean"], d["period_std"]],
            [d["ppg_entropy_mean"], d["ppg_entropy_std"]],
            d["ppg_mean_dist"],
        ]
    )
    d["feature_vec"] = vec.astype(np.float32)
    return d


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_sample_index(sample_rows, output_dir):
    """Save selected sample metadata in heatmap row order for direct listening."""
    csv_path = output_dir / "11_sample_index.csv"
    txt_path = output_dir / "11_listen_paths.txt"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "heatmap_row",
                "group",
                "utt_id",
                "rel_path",
                "audio_path",
                "prosody_score",
                "hesitation_count",
                "epenthesis_count",
                "sil_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    with open(txt_path, "w") as f:
        for row in sample_rows:
            f.write(
                f"row={row['heatmap_row']:02d}\tgroup={row['group']}\t"
                f"utt_id={row['utt_id']}\tscore={row['prosody_score']:.2f}\t"
                f"path={row['audio_path']}\n"
            )

    print(f"  Saved: {csv_path}")
    print(f"  Saved: {txt_path}")


def plot_score_distribution(scores_all, high_thresh, low_thresh, output_dir, split):
    """Histogram of prosodic scores with high/low regions annotated."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0.5, 11.5, 1)
    ax.hist(scores_all, bins=bins, color="#A8DADC", edgecolor="white", zorder=2)
    ax.axvspan(
        low_thresh + 0.5,
        -0.5,
        alpha=0.15,
        color=COLORS["low"],
        label=f"Low (≤{low_thresh})",
    )
    ax.axvspan(
        high_thresh - 0.5,
        10.5,
        alpha=0.15,
        color=COLORS["high"],
        label=f"High (≥{high_thresh})",
    )
    ax.set_xlabel("Prosodic Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Distribution of Prosodic Scores ({split} split)", fontsize=13)
    ax.legend()
    ax.set_xticks(range(0, 11))
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_dir / "01_score_distribution.png")


def _violin_or_box(ax, data_high, data_low, label, color_high, color_low):
    """Draw side-by-side violin plots for one feature scalar."""
    parts_h = ax.violinplot([data_high], positions=[0], showmedians=True)
    parts_l = ax.violinplot([data_low], positions=[1], showmedians=True)
    for pc in parts_h["bodies"]:
        pc.set_facecolor(color_high)
        pc.set_alpha(0.7)
    for pc in parts_l["bodies"]:
        pc.set_facecolor(color_low)
        pc.set_alpha(0.7)
    for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
        if key in parts_h:
            parts_h[key].set_color(color_high)
        if key in parts_l:
            parts_l[key].set_color(color_low)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["High", "Low"])
    ax.set_title(label)


def plot_scalar_features(stats_high, stats_low, output_dir):
    """Violin plots for scalar prosodic feature statistics."""
    scalar_keys = [
        ("pitch_mean", "Mean Pitch (voiced)"),
        ("pitch_std", "Pitch Std (voiced)"),
        ("pitch_range", "Pitch Range (voiced)"),
        ("voiced_ratio", "Voiced Ratio"),
        ("period_mean", "Mean Periodicity"),
        ("ppg_entropy_mean", "Mean PPG Entropy"),
    ]

    n = len(scalar_keys)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, (key, label) in enumerate(scalar_keys):
        high_vals = [s[key] for s in stats_high if key in s]
        low_vals = [s[key] for s in stats_low if key in s]
        if len(high_vals) < 2 or len(low_vals) < 2:
            axes[i].set_visible(False)
            continue
        _violin_or_box(
            axes[i], high_vals, low_vals, label, COLORS["high"], COLORS["low"]
        )
        # Mann-Whitney U significance
        stat, pval = stats.mannwhitneyu(high_vals, low_vals, alternative="two-sided")
        axes[i].set_xlabel(f"p={pval:.3f}", fontsize=9, color="gray")
        axes[i].grid(axis="y", alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    from matplotlib.patches import Patch

    legend_els = [
        Patch(facecolor=COLORS["high"], label="High"),
        Patch(facecolor=COLORS["low"], label="Low"),
    ]
    fig.legend(handles=legend_els, loc="upper right", fontsize=11)
    fig.suptitle(
        "Scalar Feature Comparison: High vs. Low Prosody Score",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, output_dir / "02_scalar_features.png")


def _mean_contour(contours, n_bins=100):
    """Resample contours to n_bins length and compute mean ± std."""
    resampled = []
    for c in contours:
        if len(c) < 2:
            continue
        x_old = np.linspace(0, 1, len(c))
        x_new = np.linspace(0, 1, n_bins)
        resampled.append(np.interp(x_new, x_old, c))
    if not resampled:
        return np.zeros(n_bins), np.zeros(n_bins)
    arr = np.stack(resampled)
    return arr.mean(axis=0), arr.std(axis=0)


def plot_contours(stats_high, stats_low, output_dir, n_bins=100):
    """Mean ± std of pitch/loudness/periodicity contours for high vs. low."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    x = np.linspace(0, 100, n_bins)

    configs = [
        ("pitch_contour", axes[0], "Pitch (Hz, voiced=non-zero)"),
        ("loudness_contour", axes[1], "Loudness (dB, mean across bands)"),
        ("period_contour", axes[2], "Periodicity"),
    ]

    for key, ax, title in configs:
        for group_label, group_stats, color in [
            ("High", stats_high, COLORS["high"]),
            ("Low", stats_low, COLORS["low"]),
        ]:
            contours = [s[key] for s in group_stats if key in s and len(s[key]) >= 2]
            mu, sigma = _mean_contour(contours, n_bins)
            ax.plot(x, mu, color=color, lw=2, label=group_label)
            ax.fill_between(x, mu - sigma, mu + sigma, color=color, alpha=0.2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Relative position in utterance (%)")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Average Feature Contours: High vs. Low Prosody Score",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_dir / "03_feature_contours.png")


def plot_loudness_bands(stats_high, stats_low, output_dir):
    """Bar chart: mean loudness per frequency band for high vs. low."""
    high_loud = np.stack([s["loudness_mean"] for s in stats_high])  # (N, 8)
    low_loud = np.stack([s["loudness_mean"] for s in stats_low])

    high_mu = high_loud.mean(axis=0)
    high_se = high_loud.std(axis=0) / np.sqrt(len(high_loud))
    low_mu = low_loud.mean(axis=0)
    low_se = low_loud.std(axis=0) / np.sqrt(len(low_loud))

    bands = np.arange(LOUDNESS_DIM)
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(
        bands - width / 2,
        high_mu,
        width,
        yerr=high_se,
        label="High",
        color=COLORS["high"],
        alpha=0.8,
        capsize=3,
    )
    ax.bar(
        bands + width / 2,
        low_mu,
        width,
        yerr=low_se,
        label="Low",
        color=COLORS["low"],
        alpha=0.8,
        capsize=3,
    )
    ax.set_xticks(bands)
    ax.set_xticklabels([f"Band {i}" for i in range(LOUDNESS_DIM)], rotation=30)
    ax.set_ylabel("Mean Loudness (dB)")
    ax.set_title("Per-Band Loudness: High vs. Low Prosody Score", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir / "04_loudness_bands.png")


def plot_ppg_distribution(stats_high, stats_low, output_dir):
    """Bar plots of mean PPG distribution (split non-silence vs silence)."""
    high_ppg = np.stack([s["ppg_mean_dist"] for s in stats_high])  # (N, 40)
    low_ppg = np.stack([s["ppg_mean_dist"] for s in stats_low])

    fig, axes = plt.subplots(
        1, 4, figsize=(22, 4), gridspec_kw={"width_ratios": [39, 39, 39, 6]}
    )

    high_mean = high_ppg.mean(axis=0)
    low_mean = low_ppg.mean(axis=0)
    diff = high_mean - low_mean
    phonemes = PPG_PHONEMES[:39]
    x_non_sil = np.arange(39)

    ax = axes[0]
    ax.bar(x_non_sil, high_mean[:39], color=COLORS["high"], alpha=0.8)
    ax.set_title("High: Mean PPG (non-silence)")
    ax.set_xlabel("Phoneme")
    ax.set_ylabel("Probability")
    ax.set_xticks(x_non_sil)
    ax.set_xticklabels(phonemes, rotation=90, ha="center", fontsize=7)

    ax = axes[1]
    ax.bar(x_non_sil, low_mean[:39], color=COLORS["low"], alpha=0.8)
    ax.set_title("Low: Mean PPG (non-silence)")
    ax.set_xlabel("Phoneme")
    ax.set_xticks(x_non_sil)
    ax.set_xticklabels(phonemes, rotation=90, ha="center", fontsize=7)

    ax = axes[2]
    colors = [COLORS["high"] if v > 0 else COLORS["low"] for v in diff[:39]]
    ax.bar(x_non_sil, diff[:39], color=colors, alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Difference (High − Low, non-silence)")
    ax.set_xlabel("Phoneme")
    ax.set_xticks(x_non_sil)
    ax.set_xticklabels(phonemes, rotation=90, ha="center", fontsize=7)

    ax = axes[3]
    sil_high = high_mean[39]
    sil_low = low_mean[39]
    sil_diff = sil_high - sil_low
    ax.bar(
        [0, 1], [sil_high, sil_low], color=[COLORS["high"], COLORS["low"]], alpha=0.85
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["High", "Low"])
    ax.set_xlabel("SIL")
    ax.set_title(f"Silence (SIL)\nΔ={sil_diff:+.4f}")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "PPG (Phonetic PosteriorGram) Comparison with SIL separated",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_dir / "05_ppg_distribution.png")


def plot_ppg_heatmap(stats_high, stats_low, output_dir):
    """Individual PPG mean distributions as a heatmap (samples × phones)."""
    high_ppg = np.stack([s["ppg_mean_dist"] for s in stats_high])
    low_ppg = np.stack([s["ppg_mean_dist"] for s in stats_low])
    combined = np.concatenate([high_ppg, low_ppg], axis=0)
    n_high = len(high_ppg)

    non_sil = combined[:, :39]  # phone index 0-38
    silence = combined[:, 39:40]  # phone index 39

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, max(6, combined.shape[0] * 0.25 + 2)),
        gridspec_kw={"width_ratios": [39, 1]},
    )

    ax_main, ax_sil = axes

    im_main = ax_main.imshow(
        non_sil, aspect="auto", cmap="viridis", interpolation="nearest"
    )
    ax_main.axhline(n_high - 0.5, color="white", lw=2, linestyle="--")
    ax_main.set_xlabel("Phoneme (non-silence)")
    ax_main.set_ylabel("Sample index")
    ax_main.set_title("PPG Average Distribution (non-silence)", fontsize=12)
    xticks_main = np.arange(39)
    ax_main.set_xticks(xticks_main)
    ax_main.set_xticklabels(
        [PPG_PHONEMES[i] for i in xticks_main], rotation=90, ha="center", fontsize=7
    )

    im_sil = ax_sil.imshow(
        silence, aspect="auto", cmap="viridis", interpolation="nearest"
    )
    ax_sil.axhline(n_high - 0.5, color="white", lw=2, linestyle="--")
    ax_sil.set_xticks([0])
    ax_sil.set_xticklabels(["SIL"])
    ax_sil.set_xlabel("Phoneme")
    ax_sil.set_title("PPG Silence", fontsize=12)

    ax_sil.text(
        1.15,
        n_high / 2,
        "HIGH",
        va="center",
        color=COLORS["high"],
        fontsize=11,
        fontweight="bold",
        transform=ax_sil.transData,
    )
    ax_sil.text(
        1.15,
        n_high + len(low_ppg) / 2,
        "LOW",
        va="center",
        color=COLORS["low"],
        fontsize=11,
        fontweight="bold",
        transform=ax_sil.transData,
    )

    cbar_main = fig.colorbar(im_main, ax=ax_main, fraction=0.025, pad=0.02)
    cbar_main.set_label("Mean probability")
    cbar_sil = fig.colorbar(im_sil, ax=ax_sil, fraction=0.15, pad=0.02)
    cbar_sil.set_label("Mean probability")

    fig.suptitle(
        "PPG Average Distribution per Sample (split silence vs non-silence)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_dir / "06_ppg_heatmap.png")


def plot_sil_run_histogram(stats_high, stats_low, output_dir):
    """Histogram of interior SIL run lengths for High vs Low with threshold bands."""
    high_runs = [l for s in stats_high for l in s.get("sil_run_lengths", [])]
    low_runs = [l for s in stats_low for l in s.get("sil_run_lengths", [])]

    if not high_runs and not low_runs:
        print(
            "  Warning: no interior SIL runs found; skipping 09_sil_run_histogram.png"
        )
        return

    max_len = max(max(high_runs, default=0), max(low_runs, default=0))
    bins = np.arange(0.5, min(max_len + 1.5, 60), 1)  # cap display at 60 frames

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, run_list, label, color in [
        (axes[0], high_runs, "High prosody", COLORS["high"]),
        (axes[1], low_runs, "Low prosody", COLORS["low"]),
    ]:
        ax.hist(run_list, bins=bins, color=color, alpha=0.75, edgecolor="white")
        # Shade epenthesis region
        ax.axvspan(
            0.5,
            EPENTHESIS_MAX_FRAMES + 0.5,
            alpha=0.12,
            color="#F4A261",
            label=f"Epenthesis (≤{EPENTHESIS_MAX_FRAMES} fr)",
        )
        # Shade hesitation region
        ax.axvspan(
            HESITATION_MIN_FRAMES - 0.5,
            bins[-1],
            alpha=0.12,
            color="#2A9D8F",
            label=f"Hesitation (≥{HESITATION_MIN_FRAMES} fr)",
        )
        ax.axvline(EPENTHESIS_MAX_FRAMES + 0.5, color="#F4A261", lw=1.5, linestyle="--")
        ax.axvline(HESITATION_MIN_FRAMES - 0.5, color="#2A9D8F", lw=1.5, linestyle="--")
        ax.set_title(f"{label}  (n={len(run_list)} runs)", fontsize=12)
        ax.set_xlabel("Run length (frames, 1 frame ≈ 16 ms)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Interior SIL Run Length Distribution\n"
        f"Epenthesis ≤{EPENTHESIS_MAX_FRAMES} fr  |  "
        f"Hesitation ≥{HESITATION_MIN_FRAMES} fr  (≈{HESITATION_MIN_FRAMES * 16} ms)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_dir / "09_sil_run_histogram.png")


def plot_linguistic_categories(stats_high, stats_low, output_dir):
    """Violin + strip plots comparing hesitation/epenthesis counts for High vs Low."""
    metrics = [
        ("hesitation_count", "Hesitation Count\n(long interior SIL runs)"),
        ("epenthesis_count", "Epenthesis Count\n(short isolated SIL bursts)"),
        ("total_sil_runs", "Total Interior\nSIL Runs"),
        ("sil_rate", "Overall SIL Rate"),
        ("mean_sil_run_len", "Mean Interior SIL\nRun Length (frames)"),
    ]

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))

    from matplotlib.patches import Patch

    rng = np.random.default_rng(0)
    for ax, (key, label) in zip(axes, metrics):
        h_vals = np.array([s.get(key, 0.0) for s in stats_high], dtype=float)
        l_vals = np.array([s.get(key, 0.0) for s in stats_low], dtype=float)

        # Violin
        if len(h_vals) >= 2:
            parts = ax.violinplot([h_vals], positions=[0], showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(COLORS["high"])
                pc.set_alpha(0.6)
            for k in ["cbars", "cmins", "cmaxes", "cmedians"]:
                if k in parts:
                    parts[k].set_color(COLORS["high"])
        if len(l_vals) >= 2:
            parts = ax.violinplot([l_vals], positions=[1], showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(COLORS["low"])
                pc.set_alpha(0.6)
            for k in ["cbars", "cmins", "cmaxes", "cmedians"]:
                if k in parts:
                    parts[k].set_color(COLORS["low"])

        # Strip (jittered dots)
        ax.scatter(
            rng.uniform(-0.12, 0.12, len(h_vals)),
            h_vals,
            c=COLORS["high"],
            alpha=0.8,
            s=25,
            zorder=3,
        )
        ax.scatter(
            1 + rng.uniform(-0.12, 0.12, len(l_vals)),
            l_vals,
            c=COLORS["low"],
            alpha=0.8,
            s=25,
            zorder=3,
        )

        # Statistics
        if len(h_vals) >= 2 and len(l_vals) >= 2:
            _, pval = stats.mannwhitneyu(h_vals, l_vals, alternative="two-sided")
            ax.set_xlabel(f"p={pval:.3f}", fontsize=9, color="gray")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["High", "Low"])
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    legend_els = [
        Patch(facecolor=COLORS["high"], label="High prosody"),
        Patch(facecolor=COLORS["low"], label="Low prosody"),
    ]
    fig.legend(handles=legend_els, loc="upper right", fontsize=10)
    fig.suptitle(
        "Linguistic Category Analysis: Hesitations vs Epenthesis/Mispronunciations\n"
        "High vs. Low Prosody Score (Mann-Whitney U p-value shown)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, output_dir / "10_linguistic_categories.png")


def plot_tsne(stats_high, stats_low, output_dir):
    """t-SNE of utterance-level feature vectors coloured by prosody group."""
    high_vecs = np.stack([s["feature_vec"] for s in stats_high])
    low_vecs = np.stack([s["feature_vec"] for s in stats_low])
    X = np.concatenate([high_vecs, low_vecs], axis=0)
    labels = np.array([1] * len(high_vecs) + [0] * len(low_vecs))

    # Remove NaN / Inf
    bad = ~np.isfinite(X).all(axis=1)
    if bad.any():
        print(
            f"  Warning: {bad.sum()} samples have NaN/Inf in feature vector, removing."
        )
        X = X[~bad]
        labels = labels[~bad]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    perplexity = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    X_2d = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7, 6))
    for grp, color, name in [(1, COLORS["high"], "High"), (0, COLORS["low"], "Low")]:
        mask = labels == grp
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=color,
            label=name,
            alpha=0.8,
            s=60,
            edgecolors="white",
            linewidths=0.5,
        )
    ax.set_title(
        "t-SNE of Handcrafted Features\nHigh vs. Low Prosody Score", fontsize=13
    )
    ax.legend(fontsize=11)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.2)
    _save(fig, output_dir / "07_tsne.png")


def plot_feature_correlation(stats_high, stats_low, output_dir):
    """Scatter plots of each scalar feature vs (0=low, 1=high) group."""
    scalar_keys = [
        ("pitch_mean", "Mean Pitch"),
        ("pitch_std", "Pitch Std"),
        ("pitch_range", "Pitch Range"),
        ("voiced_ratio", "Voiced Ratio"),
        ("period_mean", "Mean Periodicity"),
        ("ppg_entropy_mean", "PPG Entropy"),
    ]
    all_stats = stats_high + stats_low
    scores = [1.0] * len(stats_high) + [0.0] * len(stats_low)

    n = len(scalar_keys)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, (key, label) in enumerate(scalar_keys):
        vals = np.array([s.get(key, 0.0) for s in all_stats])
        sc = np.array(scores)
        finite = np.isfinite(vals)
        rho, pval = stats.spearmanr(sc[finite], vals[finite])
        c = [COLORS["high"] if s == 1.0 else COLORS["low"] for s in sc]
        axes[i].scatter(
            sc + np.random.uniform(-0.05, 0.05, len(sc)), vals, c=c, alpha=0.7, s=40
        )
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["Low", "High"])
        axes[i].set_title(f"{label}\nρ={rho:.3f} p={pval:.3f}", fontsize=10)
        axes[i].grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Feature vs. Prosody Group (Spearman ρ)", fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, output_dir / "08_feature_correlation.png")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze handcrafted features for high vs. low prosody score samples"
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("data/speechocean762/so762"),
        help="Root dir containing train/ and test/ subdirs with wav.scp",
    )
    parser.add_argument(
        "--label_dir",
        type=Path,
        default=Path("data/speechocean762"),
        help="Directory containing te_label_utt.npy / tr_label_utt.npy and *_handcrafted_feats.pkl",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("exp/handcrafted_analysis"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Which split to analyse",
    )
    parser.add_argument(
        "--n_samples", type=int, default=30, help="Max samples per group (high / low)"
    )
    parser.add_argument(
        "--high_thresh",
        type=float,
        default=9.0,
        help="Minimum prosodic score to be in the 'high' group",
    )
    parser.add_argument(
        "--low_thresh",
        type=float,
        default=5.0,
        help="Maximum prosodic score to be in the 'low' group",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prefix = "tr" if args.split == "train" else "te"
    label_path = args.label_dir / f"{prefix}_label_utt.npy"
    feats_path = args.label_dir / f"{prefix}_handcrafted_feats.pkl"
    wav_scp = args.dataset_dir / args.split / "wav.scp"

    # ------------------------------------------------------------------
    # 1. Load labels, wav paths, and pre-extracted features
    # ------------------------------------------------------------------
    print(f"\n[1/3] Loading labels from {label_path}")
    labels = np.load(label_path)  # (N, 5)
    prosodic_scores = labels[:, PROSODIC_IDX]

    print(f"[1/3] Loading wav.scp from {wav_scp}")
    utt_entries = load_wav_scp(wav_scp)  # list of (utt_id, rel_path)
    assert len(utt_entries) == len(labels), (
        f"wav.scp has {len(utt_entries)} entries but labels has {len(labels)}"
    )

    print(f"[1/3] Loading pre-extracted features from {feats_path}")
    with open(feats_path, "rb") as f:
        feats = pickle.load(f)
    print(f"       {len(feats)} utterances in pkl")

    # ------------------------------------------------------------------
    # 2. Select high / low samples
    # ------------------------------------------------------------------
    print(
        f"\n[2/3] Selecting samples "
        f"(high ≥ {args.high_thresh}, low ≤ {args.low_thresh})"
    )

    high_idx = np.where(prosodic_scores >= args.high_thresh)[0]
    low_idx = np.where(prosodic_scores <= args.low_thresh)[0]

    rng = np.random.default_rng(42)
    high_idx = rng.choice(
        high_idx, size=min(args.n_samples, len(high_idx)), replace=False
    )
    low_idx = rng.choice(low_idx, size=min(args.n_samples, len(low_idx)), replace=False)

    print(
        f"  High group: {len(high_idx)} samples  "
        f"(scores {prosodic_scores[high_idx].mean():.2f} ± "
        f"{prosodic_scores[high_idx].std():.2f})"
    )
    print(
        f"  Low  group: {len(low_idx)} samples  "
        f"(scores {prosodic_scores[low_idx].mean():.2f} ± "
        f"{prosodic_scores[low_idx].std():.2f})"
    )

    # ------------------------------------------------------------------
    # 3. Compute stats from pre-extracted features
    # ------------------------------------------------------------------
    print(f"\n[3/3] Computing feature statistics …")
    all_indices = np.concatenate([high_idx, low_idx])
    cache = {}
    sample_meta = {}
    for idx in tqdm(all_indices, desc="Computing stats"):
        utt_id, rel_path = utt_entries[idx]
        loud, pitch, period, ppg = load_feats_from_pkl(feats, rel_path)
        if loud is None:
            print(f"\n  Warning: {utt_id} ({rel_path}) not found in pkl, skipping.")
            cache[idx] = None
        else:
            cache[idx] = compute_stats(loud, pitch, period, ppg)
            sample_meta[idx] = {
                "utt_id": utt_id,
                "rel_path": rel_path,
                "audio_path": str((args.dataset_dir / args.split / rel_path).resolve()),
                "prosody_score": float(prosodic_scores[idx]),
            }

    stats_high = [cache[i] for i in high_idx if cache.get(i) is not None]
    stats_low = [cache[i] for i in low_idx if cache.get(i) is not None]
    print(f"  Valid samples — High: {len(stats_high)}, Low: {len(stats_low)}")

    sample_rows = []
    row_idx = 0
    for group_name, group_indices in [("high", high_idx), ("low", low_idx)]:
        for idx in group_indices:
            if cache.get(idx) is None:
                continue
            meta = sample_meta[idx]
            stat = cache[idx]
            sample_rows.append(
                {
                    "heatmap_row": row_idx,
                    "group": group_name,
                    "utt_id": meta["utt_id"],
                    "rel_path": meta["rel_path"],
                    "audio_path": meta["audio_path"],
                    "prosody_score": meta["prosody_score"],
                    "hesitation_count": stat.get("hesitation_count", 0),
                    "epenthesis_count": stat.get("epenthesis_count", 0),
                    "sil_rate": stat.get("sil_rate", 0.0),
                }
            )
            row_idx += 1

    save_sample_index(sample_rows, args.output_dir)

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------
    print(f"\nGenerating plots → {args.output_dir}/")

    plot_score_distribution(
        prosodic_scores, args.high_thresh, args.low_thresh, args.output_dir, args.split
    )
    plot_scalar_features(stats_high, stats_low, args.output_dir)
    plot_contours(stats_high, stats_low, args.output_dir)
    plot_loudness_bands(stats_high, stats_low, args.output_dir)
    plot_ppg_distribution(stats_high, stats_low, args.output_dir)
    plot_ppg_heatmap(stats_high, stats_low, args.output_dir)
    plot_tsne(stats_high, stats_low, args.output_dir)
    plot_feature_correlation(stats_high, stats_low, args.output_dir)
    plot_sil_run_histogram(stats_high, stats_low, args.output_dir)
    plot_linguistic_categories(stats_high, stats_low, args.output_dir)

    print(f"\nDone! All figures saved to {args.output_dir.resolve()}/")
    print("\nFigure summary:")
    print("  01_score_distribution.png  — score histogram with group regions")
    print("  02_scalar_features.png     — violin plots for scalar feature statistics")
    print(
        "  03_feature_contours.png    — mean ± std pitch/loudness/periodicity over time"
    )
    print("  04_loudness_bands.png      — per-band loudness bar chart")
    print("  05_ppg_distribution.png    — mean PPG phone distribution + diff")
    print("  06_ppg_heatmap.png         — per-sample PPG heatmap")
    print("  07_tsne.png                — t-SNE of handcrafted feature vectors")
    print("  08_feature_correlation.png — feature vs. group scatter + Spearman ρ")
    print(
        "  09_sil_run_histogram.png   — interior SIL run length distribution (epenthesis vs hesitation regions)"
    )
    print(
        "  10_linguistic_categories.png — violin+strip: hesitation/epenthesis counts, SIL rate, run length"
    )
    print("  11_sample_index.csv        — heatmap row to utt_id/audio path mapping")
    print("  11_listen_paths.txt        — compact sample list for direct listening")


if __name__ == "__main__":
    main()
