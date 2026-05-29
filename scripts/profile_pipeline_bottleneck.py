#!/usr/bin/env python3
"""Profile the whole training pipeline with stage-wise timing.

Current focus: FDMPAScorer full train-step pipeline.
Measures per-step latency breakdown:
- data_wait (includes on-the-fly feature extraction in collate)
- to_device
- mine_update
- forward_main
- loss_compute
- backward
- optimizer_step
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from prosody_scorer.models import FDMPAScorer
from prosody_scorer.speech_datasets import OnTheFlyFeatureCollator, create_dataset
from prosody_scorer.train import unpack_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-type", type=str, default="so762")
    parser.add_argument("--data-dir", type=str, default="data/speechocean762")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--aspect", nargs="+", default=["fluency"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--ssl-dim", type=int, default=1024)
    parser.add_argument("--mine-hidden-dim", type=int, default=64)
    parser.add_argument("--mine-ema-decay", type=float, default=0.99)
    parser.add_argument("--mine-steps", type=int, default=1)
    parser.add_argument("--mine-weight", type=float, default=0.01)
    parser.add_argument("--mi-neg-penalty-weight", type=float, default=0.5)
    parser.add_argument("--mi-main-scale-max", type=float, default=50.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-duration-sec", type=float, default=30.0)
    parser.add_argument("--collate-timing", action="store_true")
    parser.add_argument("--collate-timing-report-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=66)
    return parser.parse_args()


def resolve_device(device_flag: str) -> torch.device:
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available")
        return torch.device("cuda")
    if device_flag == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(device: torch.device, fn):
    sync_if_cuda(device)
    t0 = time.perf_counter()
    out = fn()
    sync_if_cuda(device)
    return out, (time.perf_counter() - t0) * 1000.0


def fetch_next_batch(it, loader):
    try:
        batch = next(it)
    except StopIteration:
        it = iter(loader)
        batch = next(it)
    return it, batch


def build_dataloader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    # FDMPAScorer requires HC features, so force fdmpa feature mode.
    feature_type = "fdmpa"
    dataset = create_dataset(
        dataset_type=args.dataset_type,
        split=args.train_split,
        aspects=args.aspect,
        device=device,
        data_dir=args.data_dir,
        feature_type=feature_type,
        dataset_name=args.dataset_type,
        max_duration_sec=args.max_duration_sec,
        on_the_fly_features=True,
    )
    collate_fn = OnTheFlyFeatureCollator(
        feature_type=feature_type,
        device=device,
        kmeans_model=None,
        timing_enabled=args.collate_timing,
        timing_report_every=args.collate_timing_report_every,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )


def build_model(args: argparse.Namespace, device: torch.device) -> FDMPAScorer:
    model = FDMPAScorer(
        ssl_input_dim=args.ssl_dim,
        hidden_dim=args.hidden_dim,
        scorers=args.aspect,
        mine_hidden_dim=args.mine_hidden_dim,
        mine_ema_decay=args.mine_ema_decay,
        num_tokens=-1,
        dropout_prob=0.1,
    )
    return model.to(device)


def summarize_stage_times(times: dict[str, list[float]], steps: int) -> None:
    totals = {k: float(sum(v)) for k, v in times.items() if v}
    total_ms = sum(totals.values())

    print("\n=== Whole-pipeline stage timing ===")
    print(f"Profiled steps: {steps}")
    print(f"Total profiled time: {total_ms:.2f} ms")
    print(f"Average step latency: {total_ms / max(1, steps):.2f} ms")
    print(
        f"{'stage':<16} {'total_ms':>12} {'avg_ms':>12} {'p50_ms':>12} {'p90_ms':>12} {'share':>8}"
    )

    for stage, vals in sorted(totals.items(), key=lambda x: x[1], reverse=True):
        series = times[stage]
        avg_ms = float(np.mean(series))
        p50_ms = float(np.percentile(series, 50))
        p90_ms = float(np.percentile(series, 90))
        share = 100.0 * vals / max(1e-9, total_ms)
        print(
            f"{stage:<16} {vals:>12.2f} {avg_ms:>12.2f} {p50_ms:>12.2f} {p90_ms:>12.2f} {share:>7.2f}%"
        )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    print("Whole pipeline bottleneck profiling")
    print(f"Device: {device}")
    # Inform penn about preferred GPU (if any)
    try:
        import penn as _penn
        if device.type == 'cuda':
            _penn._DEFAULT_GPU = device.index if device.index is not None else 0
        else:
            _penn._DEFAULT_GPU = None
    except Exception:
        pass
    print(
        f"Dataset={args.dataset_type}, split={args.train_split}, batch={args.batch_size}, workers={args.num_workers}"
    )

    loader = build_dataloader(args, device)
    model = build_model(args, device)
    loss_fn = nn.MSELoss()

    main_params = [p for p in model.non_mine_parameters() if p.requires_grad]
    mine_params = [p for p in model.mine_parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(main_params, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    mine_optimizer = torch.optim.Adam(
        mine_params,
        args.lr * 0.02,
        weight_decay=5e-7,
        betas=(0.95, 0.999),
    )

    stage_times: dict[str, list[float]] = defaultdict(list)
    full_step_times = []

    it = iter(loader)
    total_loops = args.warmup_steps + args.max_steps
    for step in range(total_loops):
        t_step_start = time.perf_counter()

        sync_if_cuda(device)
        t0 = time.perf_counter()
        it, batch = fetch_next_batch(it, loader)
        sync_if_cuda(device)
        t_data_wait = (time.perf_counter() - t0) * 1000.0

        _, _, feats, hc_feats, _ = unpack_batch(batch)

        sync_if_cuda(device)
        t0 = time.perf_counter()
        feats_dev = feats.to(device)
        hc_dev = hc_feats.to(device)
        sync_if_cuda(device)
        t_to_device = (time.perf_counter() - t0) * 1000.0

        sync_if_cuda(device)
        t0 = time.perf_counter()
        for _ in range(max(1, args.mine_steps)):
            with torch.no_grad():
                _, aux_detached = model(feats_dev, hc_dev)
            mi_local, _, _, _ = model.compute_mi_terms(
                aux_detached["ssl_tokens"],
                aux_detached["hc_tokens"],
                aux_detached["ssl_global"],
                aux_detached["hc_global"],
                update_ema=True,
            )
            mi_local_for_mine = torch.clamp(mi_local, min=-5.0, max=5.0)
            mi_neg_penalty = F.relu(-mi_local_for_mine)
            mine_loss = (-(mi_local_for_mine) + args.mi_neg_penalty_weight * mi_neg_penalty)
            if not torch.isfinite(mine_loss):
                continue
            mine_optimizer.zero_grad()
            mine_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.mine_parameters(), max_norm=5.0)
            mine_optimizer.step()
        sync_if_cuda(device)
        t_mine_update = (time.perf_counter() - t0) * 1000.0

        sync_if_cuda(device)
        t0 = time.perf_counter()
        pred, aux = model(feats_dev, hc_dev)
        sync_if_cuda(device)
        t_forward_main = (time.perf_counter() - t0) * 1000.0

        labels = batch[1]
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        sync_if_cuda(device)
        t0 = time.perf_counter()
        labels_dev = labels.to(device, non_blocking=True)
        mi_local, _, _, _ = model.compute_mi_terms(
            aux["ssl_tokens"],
            aux["hc_tokens"],
            aux["ssl_global"],
            aux["hc_global"],
            update_ema=False,
        )
        if not torch.isfinite(mi_local):
            mi_local = torch.zeros((), device=device)
        branch_aux_loss = torch.zeros((), device=device)
        for branch_pred in aux["branch_preds"].values():
            branch_aux_loss = branch_aux_loss + loss_fn(branch_pred, labels_dev)
        branch_aux_loss = branch_aux_loss / max(1, len(aux["branch_preds"]))

        mse_loss = loss_fn(pred, labels_dev)
        mi_neg_penalty_main = F.relu(-mi_local)
        mi_main_term = args.mine_weight * mi_local
        mi_main_scale = torch.clamp(
            mse_loss.detach() / (mi_main_term.detach().abs() + 1e-6),
            min=1.0,
            max=args.mi_main_scale_max,
        )
        loss = (
            mse_loss
            - mi_main_scale * mi_main_term
            + args.mi_neg_penalty_weight * mi_neg_penalty_main
            + 0.1 * branch_aux_loss
        )
        sync_if_cuda(device)
        t_loss_compute = (time.perf_counter() - t0) * 1000.0

        sync_if_cuda(device)
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        sync_if_cuda(device)
        t_backward = (time.perf_counter() - t0) * 1000.0

        _, t_optimizer_step = timed_call(device, optimizer.step)

        sync_if_cuda(device)
        step_ms = (time.perf_counter() - t_step_start) * 1000.0

        # Skip warmup when collecting statistics.
        if step >= args.warmup_steps:
            stage_times["data_wait"].append(t_data_wait)
            stage_times["to_device"].append(t_to_device)
            stage_times["mine_update"].append(t_mine_update)
            stage_times["forward_main"].append(t_forward_main)
            stage_times["loss_compute"].append(t_loss_compute)
            stage_times["backward"].append(t_backward)
            stage_times["optimizer_step"].append(t_optimizer_step)
            full_step_times.append(step_ms)

        if (step + 1) % 2 == 0:
            phase = "warmup" if step < args.warmup_steps else "profile"
            print(
                f"step={step + 1}/{total_loops} ({phase}) total_step_ms={step_ms:.2f} "
                f"data={t_data_wait:.2f} fwd={t_forward_main:.2f} bwd={t_backward:.2f}"
            )

    summarize_stage_times(stage_times, args.max_steps)
    if full_step_times:
        print("\n=== End-to-end step latency summary ===")
        print(f"mean={np.mean(full_step_times):.2f} ms")
        print(f"p50={np.percentile(full_step_times, 50):.2f} ms")
        print(f"p90={np.percentile(full_step_times, 90):.2f} ms")
        print(f"p99={np.percentile(full_step_times, 99):.2f} ms")


if __name__ == "__main__":
    main()
