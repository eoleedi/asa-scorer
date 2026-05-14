#!/usr/bin/env python3
"""Compute speaker association and residual speaker bias for prosody scores."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from prosody_scorer.models import ClusterScorer, FDMPAScorer
from prosody_scorer.speech_datasets import (
    create_dataset,
    custom_collate_fn,
    fdmpa_collate_fn,
)


DEFAULT_FDMPA_CKPT = (
    "exp/SpeechOcean762/FDMPAScorer_prosodic_MINE-2stage/"
    "rolling-marginal/models/best_audio_model.pth"
)
DEFAULT_BASELINE_CKPT = (
    "exp/SpeechOcean762/ClusterScorer_fluency+prosodic_baseline/"
    "1e-3-3-25-32-ClusterScorer-br/0/models/best_audio_model.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how much score variance and prediction-error variance are "
            "associated with speaker ID. Speaker ID is categorical, so the script "
            "reports correlation ratio eta and eta^2."
        )
    )
    parser.add_argument("--data-dir", default="data/speechocean762")
    parser.add_argument("--hf-dataset", default="mispeech/speechocean762")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--speaker-column", default="speaker")
    parser.add_argument("--aspect", default="prosodic")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fdmpa-checkpoint", default=DEFAULT_FDMPA_CKPT)
    parser.add_argument("--baseline-checkpoint", default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--fdmpa-hidden-dim", type=int, default=64)
    parser.add_argument("--fdmpa-num-tokens", type=int, default=-1)
    parser.add_argument("--baseline-hidden-dim", type=int, default=32)
    parser.add_argument("--baseline-num-clusters", type=int, default=50)
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to save the matrix as CSV.",
    )
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=True)


def path_to_hf_index(path: str) -> int:
    """Convert wav/test_000123.wav or test_000123 to 123."""
    stem = Path(path).stem
    return int(stem.split("_")[-1])


def get_hf_speakers(args: argparse.Namespace) -> np.ndarray:
    dataset = load_dataset(args.hf_dataset, split=args.split, cache_dir=args.hf_cache_dir)
    if args.speaker_column not in dataset.column_names:
        raise ValueError(
            f"Speaker column {args.speaker_column!r} not found. "
            f"Available columns: {dataset.column_names}"
        )
    return np.asarray(dataset[args.speaker_column]).astype(str)


def predict_fdmpa(args: argparse.Namespace, hf_speakers: np.ndarray):
    dataset = create_dataset(
        "so762",
        split=args.split,
        aspects=[args.aspect],
        data_dir=args.data_dir,
        feature_type="fdmpa",
        device=args.device,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=fdmpa_collate_fn,
    )
    model = FDMPAScorer(
        ssl_input_dim=1024,
        hidden_dim=args.fdmpa_hidden_dim,
        scorers=[args.aspect],
        num_tokens=args.fdmpa_num_tokens,
    )
    load_checkpoint(model, args.fdmpa_checkpoint, args.device)
    model.to(args.device)
    model.eval()

    preds, targets, speakers = [], [], []
    with torch.no_grad():
        for paths, labels, feats, hc_feats, _ in loader:
            pred, _ = model(feats.to(args.device), hc_feats.to(args.device))
            preds.append(pred.cpu().numpy()[:, 0])
            targets.append(labels.numpy()[:, 0])
            speakers.extend(hf_speakers[path_to_hf_index(path)] for path in paths)

    return np.concatenate(preds), np.concatenate(targets), np.asarray(speakers)


def predict_baseline(args: argparse.Namespace, hf_speakers: np.ndarray):
    aspects = ["fluency", args.aspect] if args.aspect != "fluency" else ["fluency"]
    aspect_index = aspects.index(args.aspect)
    dataset = create_dataset(
        "so762",
        split=args.split,
        aspects=aspects,
        data_dir=args.data_dir,
        feature_type="ssl",
        device=args.device,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )
    model = ClusterScorer(
        input_dim=1024,
        embed_dim=args.baseline_hidden_dim,
        scorers=aspects,
        clustering_dim=6,
        num_clusters=args.baseline_num_clusters,
    )
    load_checkpoint(model, args.baseline_checkpoint, args.device)
    model.to(args.device)
    model.eval()

    preds, targets, speakers = [], [], []
    with torch.no_grad():
        for paths, labels, feats, cluster_idx in loader:
            pred = model(feats.to(args.device), (cluster_idx + 1).to(args.device))
            preds.append(pred.cpu().numpy()[:, aspect_index])
            targets.append(labels.numpy()[:, aspect_index])
            speakers.extend(hf_speakers[path_to_hf_index(path)] for path in paths)

    return np.concatenate(preds), np.concatenate(targets), np.asarray(speakers)


def encode_speakers(speakers: np.ndarray):
    unique, inverse = np.unique(speakers, return_inverse=True)
    counts = np.bincount(inverse)
    return unique, inverse, counts


def eta_squared(values: np.ndarray, inverse: np.ndarray, counts: np.ndarray):
    values = np.asarray(values, dtype=float)
    grand_mean = values.mean()
    speaker_means = np.bincount(inverse, weights=values) / counts
    ss_between = np.sum(counts * (speaker_means - grand_mean) ** 2)
    ss_total = np.sum((values - grand_mean) ** 2)
    if ss_total == 0:
        return float("nan"), speaker_means
    return float(ss_between / ss_total), speaker_means


def adjusted_r2(r2: float, n_items: int, n_speakers: int) -> float:
    predictors = n_speakers - 1
    denominator = n_items - predictors - 1
    if denominator <= 0:
        return float("nan")
    return float(1 - (1 - r2) * (n_items - 1) / denominator)


def pooled_within_speaker_sd(
    values: np.ndarray, inverse: np.ndarray, n_speakers: int
) -> float:
    variances = []
    for speaker_idx in range(n_speakers):
        speaker_values = values[inverse == speaker_idx]
        if len(speaker_values) > 1:
            variances.append(np.var(speaker_values, ddof=1))
    return float(math.sqrt(np.mean(variances))) if variances else float("nan")


def summarize(
    name: str,
    values: np.ndarray,
    speakers: np.ndarray,
    target: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    unique, inverse, counts = encode_speakers(speakers)
    r2, speaker_means = eta_squared(values, inverse, counts)
    row: dict[str, float | int | str] = {
        "score": name,
        "n": len(values),
        "speakers": len(unique),
        "min_per_speaker": int(counts.min()),
        "max_per_speaker": int(counts.max()),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "speaker_eta": math.sqrt(r2),
        "speaker_eta2": r2,
        "speaker_adjusted_r2": adjusted_r2(r2, len(values), len(unique)),
        "speaker_mean_sd": float(np.std(speaker_means, ddof=1)),
        "speaker_mean_range": float(speaker_means.max() - speaker_means.min()),
        "within_speaker_sd_pooled": pooled_within_speaker_sd(
            values, inverse, len(unique)
        ),
    }
    if target is not None:
        residual = values - target
        residual_r2, residual_means = eta_squared(residual, inverse, counts)
        row.update(
            {
                "pcc_with_ground_truth": float(np.corrcoef(values, target)[0, 1]),
                "mae": float(np.mean(np.abs(residual))),
                "mean_error": float(np.mean(residual)),
                "residual_speaker_eta": math.sqrt(residual_r2),
                "residual_speaker_eta2": residual_r2,
                "residual_speaker_adjusted_r2": adjusted_r2(
                    residual_r2, len(values), len(unique)
                ),
                "speaker_residual_sd": float(np.std(residual_means, ddof=1)),
                "speaker_residual_range": float(
                    residual_means.max() - residual_means.min()
                ),
            }
        )
    return row


def format_value(value: float | int | str) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_matrix(rows: list[dict[str, float | int | str]]) -> None:
    columns = [
        "score",
        "speaker_eta",
        "speaker_eta2",
        "speaker_adjusted_r2",
        "residual_speaker_eta",
        "residual_speaker_eta2",
        "residual_speaker_adjusted_r2",
        "pcc_with_ground_truth",
        "mae",
    ]
    available_columns = [
        col for col in columns if any(col in row for row in rows)
    ]
    print("| " + " | ".join(available_columns) + " |")
    print("| " + " | ".join("---" for _ in available_columns) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(format_value(row.get(col, "")) for col in available_columns)
            + " |"
        )


def save_csv(rows: list[dict[str, float | int | str]], output_path: str) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    hf_speakers = get_hf_speakers(args)

    fdmpa_pred, fdmpa_target, fdmpa_speakers = predict_fdmpa(args, hf_speakers)
    baseline_pred, baseline_target, baseline_speakers = predict_baseline(args, hf_speakers)

    rows = [
        summarize(f"Ground truth {args.aspect}", fdmpa_target, fdmpa_speakers),
        summarize(
            f"FDMPA-MINE {args.aspect}", fdmpa_pred, fdmpa_speakers, fdmpa_target
        ),
        summarize(
            f"ClusterScorer baseline {args.aspect}",
            baseline_pred,
            baseline_speakers,
            baseline_target,
        ),
    ]

    print_matrix(rows)
    if args.output_csv:
        save_csv(rows, args.output_csv)
        print(f"\nSaved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
