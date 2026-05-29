#!/usr/bin/env python3
"""Forward-pass bottleneck profiler for prosody scorer models.

This script focuses on forward timing only and reports:
1) Module-level timing via forward hooks
2) Operator-level timing via torch.profiler

Use synthetic inputs by default to isolate model compute from dataloading.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import torch
from torch import nn
from torch.profiler import ProfilerActivity

from prosody_scorer.models import (
    ClusterScorer,
    CrossAttnHCSSLScorer,
    FDMPAScorer,
    LayerWeightedHCSSLScorer,
    NonClusterScorer,
    SimpleRegressionScorer,
    SingleLayerHCSSLScorer,
    TransformerScorer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default="FDMPAScorer",
        choices=[
            "ClusterScorer",
            "CrossAttnHCSSLScorer",
            "FDMPAScorer",
            "LayerWeightedHCSSLScorer",
            "NonClusterScorer",
            "SimpleRegressionScorer",
            "SingleLayerHCSSLScorer",
            "TransformerScorer",
        ],
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--ssl-dim", type=int, default=1024)
    parser.add_argument("--hc-dim", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-clusters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--topk", type=int, default=20)
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


def build_model(args: argparse.Namespace) -> nn.Module:
    scorers = ["fluency"]
    if args.model == "ClusterScorer":
        return ClusterScorer(
            input_dim=args.ssl_dim,
            embed_dim=args.hidden_dim,
            clustering_dim=6,
            num_clusters=args.num_clusters,
            scorers=scorers,
        )
    if args.model == "NonClusterScorer":
        return NonClusterScorer(
            input_dim=args.ssl_dim,
            embed_dim=args.hidden_dim,
            scorers=scorers,
        )
    if args.model == "SimpleRegressionScorer":
        return SimpleRegressionScorer(
            input_dim=args.ssl_dim,
            hidden_dim=args.hidden_dim,
            scorers=scorers,
        )
    if args.model == "TransformerScorer":
        return TransformerScorer(
            input_dim=args.ssl_dim,
            dropout_prob=0.1,
            num_heads=args.num_heads,
            depth=args.depth,
            hidden_dim=args.hidden_dim,
            clustering_dim=6,
            scorers=scorers,
        )
    if args.model == "FDMPAScorer":
        return FDMPAScorer(
            ssl_input_dim=args.ssl_dim,
            hidden_dim=args.hidden_dim,
            scorers=scorers,
            num_tokens=-1,
            dropout_prob=0.1,
        )
    if args.model == "CrossAttnHCSSLScorer":
        return CrossAttnHCSSLScorer(
            ssl_input_dim=args.ssl_dim,
            hc_input_dim=args.hc_dim,
            hidden_dim=args.hidden_dim,
            scorers=scorers,
            num_heads=args.num_heads,
            depth=args.depth,
            dropout_prob=0.1,
        )
    if args.model == "SingleLayerHCSSLScorer":
        return SingleLayerHCSSLScorer(
            ssl_input_dim=args.ssl_dim,
            hidden_dim=args.hidden_dim,
            scorers=scorers,
            num_tokens=-1,
            dropout_prob=0.1,
        )
    if args.model == "LayerWeightedHCSSLScorer":
        return LayerWeightedHCSSLScorer(
            ssl_input_dim=args.ssl_dim,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            scorers=scorers,
            num_tokens=-1,
            dropout_prob=0.1,
            hc_aux_weight=0.1,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def build_inputs(args: argparse.Namespace, device: torch.device) -> tuple:
    b, t, d = args.batch_size, args.seq_len, args.ssl_dim
    ssl_feats = torch.randn(b, t, d, device=device)
    hc_feats = torch.randn(b, t, args.hc_dim, device=device)
    cluster_ids = torch.randint(
        low=1, high=args.num_clusters + 1, size=(b, t), device=device
    )

    # Make sequence lengths variable by right-padding with zeros.
    lengths = torch.randint(
        low=max(8, t // 3), high=t + 1, size=(b,), device=device
    )
    for i in range(b):
        valid = int(lengths[i].item())
        if valid < t:
            ssl_feats[i, valid:] = 0
            hc_feats[i, valid:] = 0
            cluster_ids[i, valid:] = 0

    if args.model in ["FDMPAScorer", "CrossAttnHCSSLScorer", "SingleLayerHCSSLScorer"]:
        return (ssl_feats, hc_feats)
    if args.model == "LayerWeightedHCSSLScorer":
        ssl_layers = torch.randn(b, args.num_layers, t, d, device=device)
        for i in range(b):
            valid = int(lengths[i].item())
            if valid < t:
                ssl_layers[i, :, valid:] = 0
        return (ssl_layers, hc_feats)
    if args.model in ["ClusterScorer", "TransformerScorer"]:
        return (ssl_feats, cluster_ids)
    if args.model in ["NonClusterScorer", "SimpleRegressionScorer"]:
        return (ssl_feats,)
    raise ValueError(f"Unsupported model input for: {args.model}")


def run_forward(model: nn.Module, model_inputs: tuple):
    out = model(*model_inputs)
    if isinstance(out, tuple):
        return out[0]
    return out


def profile_modules(
    model: nn.Module,
    model_inputs: tuple,
    device: torch.device,
    warmup: int,
    steps: int,
    topk: int,
) -> None:
    time_sums_ms = defaultdict(float)
    call_counts = defaultdict(int)
    start_cache = {}
    handles = []

    def pre_hook(mod: nn.Module, _args):
        mod_id = id(mod)
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            start_cache[mod_id] = start
        else:
            start_cache[mod_id] = time.perf_counter_ns()

    def post_hook(mod: nn.Module, _args, _output):
        mod_id = id(mod)
        start = start_cache.pop(mod_id, None)
        if start is None:
            return
        if device.type == "cuda":
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize(device)
            elapsed_ms = start.elapsed_time(end)
        else:
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6

        name = mod.__class__.__name__
        time_sums_ms[name] += float(elapsed_ms)
        call_counts[name] += 1

    # Leaf modules reduce double-counting and keep output readable.
    for m in model.modules():
        if len(list(m.children())) == 0:
            handles.append(m.register_forward_pre_hook(pre_hook))
            handles.append(m.register_forward_hook(post_hook))

    try:
        with torch.inference_mode():
            for _ in range(warmup):
                _ = run_forward(model, model_inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            t0 = time.perf_counter()
            for _ in range(steps):
                _ = run_forward(model, model_inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        for h in handles:
            h.remove()

    rows = []
    for name, t_ms in time_sums_ms.items():
        calls = call_counts[name]
        rows.append((name, t_ms, calls, t_ms / max(1, calls)))
    rows.sort(key=lambda x: x[1], reverse=True)

    print("\n=== Module-level forward timing (leaf modules, aggregated over all steps) ===")
    print(f"Total measured wall time (forward only): {total_ms:.2f} ms for {steps} steps")
    print(f"Average per-step forward latency: {total_ms / max(1, steps):.2f} ms")
    print(
        f"{'rank':>4} {'module':<34} {'total_ms':>12} {'calls':>10} {'avg_ms':>12} {'share':>8}"
    )

    measured_sum = sum(x[1] for x in rows)
    for idx, (name, t_ms, calls, avg_ms) in enumerate(rows[:topk], start=1):
        share = 100.0 * t_ms / max(1e-9, measured_sum)
        print(f"{idx:>4} {name:<34} {t_ms:>12.2f} {calls:>10} {avg_ms:>12.4f} {share:>7.2f}%")


def profile_ops(
    model: nn.Module,
    model_inputs: tuple,
    device: torch.device,
    warmup: int,
    steps: int,
    topk: int,
) -> None:
    acts = [ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = run_forward(model, model_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        with torch.profiler.profile(
            activities=acts,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for _ in range(steps):
                _ = run_forward(model, model_inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    print("\n=== Operator-level timing (torch.profiler) ===")
    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=topk)
    print(table)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    model = build_model(args).to(device).eval()
    model_inputs = build_inputs(args, device)

    print("Forward bottleneck profiling")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Input setup: batch={args.batch_size}, seq_len={args.seq_len}, ssl_dim={args.ssl_dim}")
    if args.model == "LayerWeightedHCSSLScorer":
        print(f"Extra setup: num_layers={args.num_layers}")

    profile_modules(
        model=model,
        model_inputs=model_inputs,
        device=device,
        warmup=args.warmup,
        steps=args.steps,
        topk=args.topk,
    )
    profile_ops(
        model=model,
        model_inputs=model_inputs,
        device=device,
        warmup=max(2, args.warmup // 2),
        steps=max(5, args.steps // 2),
        topk=args.topk,
    )


if __name__ == "__main__":
    main()
