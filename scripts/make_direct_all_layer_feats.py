#!/usr/bin/env python3
"""Create a direct path -> tensor pickle for all-layer HuBERT features.

The current all-layer artifacts in data/speechocean762 are stored as small
pickle indexes pointing to per-utterance tensor shards. This script rebuilds a
pickle with the same shape as the old tr_feats.pkl/te_feats.pkl files:

    {"wav/train_000000.wav": torch.Tensor[L, T, D], ...}

Example:
    uv run python scripts/make_direct_all_layer_feats.py \
        --data_dir data/speechocean762 \
        --split train

To overwrite the indexed file intentionally:
    uv run python scripts/make_direct_all_layer_feats.py \
        --data_dir data/speechocean762 \
        --split train \
        --output data/speechocean762/tr_all_layer_feats.pkl \
        --overwrite-output
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


def _prefix_for_split(split: str) -> str:
    return "tr" if split == "train" else "te"


def _default_index_path(data_dir: Path, prefix: str) -> Path:
    per_utt_index = data_dir / f"{prefix}_all_layer_feats.pkl.per_utt.bak"
    if per_utt_index.exists():
        return per_utt_index
    return data_dir / f"{prefix}_all_layer_feats.pkl"


def _default_output_path(data_dir: Path, prefix: str) -> Path:
    return data_dir / f"{prefix}_all_layer_feats_direct.pkl"


def _load_index(index_path: Path) -> dict[str, Any]:
    with index_path.open("rb") as f:
        index = pickle.load(f)
    if not isinstance(index, dict):
        raise TypeError(f"Expected {index_path} to contain a dict, got {type(index)}")
    return index


def _resolve_tensor_path(data_dir: Path, entry: Any) -> Path:
    if not isinstance(entry, dict) or "tensor_path" not in entry:
        raise ValueError(
            "This converter needs a per-utterance shard index with entries like "
            "{'tensor_path': 'tr_all_layer_feats.d/<hash>.pt'}. "
            "Use the existing *.pkl.per_utt.bak file as --index."
        )

    tensor_path = Path(entry["tensor_path"])
    if not tensor_path.is_absolute():
        tensor_path = data_dir / tensor_path
    return tensor_path


def build_direct_pickle(
    data_dir: Path,
    index_path: Path,
    output_path: Path,
    overwrite_output: bool,
    max_items: int | None,
    dry_run: bool,
) -> None:
    if output_path.exists() and not overwrite_output and not dry_run:
        raise FileExistsError(
            f"{output_path} already exists. Pass --overwrite-output to replace it."
        )

    index = _load_index(index_path)
    items = list(index.items())
    if max_items is not None:
        items = items[:max_items]

    direct: dict[str, torch.Tensor] = {}
    total_bytes = 0

    for audio_path, entry in tqdm(items, desc="Loading all-layer feature shards"):
        tensor_path = _resolve_tensor_path(data_dir, entry)
        if not tensor_path.exists():
            raise FileNotFoundError(f"Missing tensor shard for {audio_path}: {tensor_path}")

        tensor = torch.load(tensor_path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected tensor in {tensor_path}, got {type(tensor)}")
        if tensor.dim() != 3:
            raise ValueError(
                f"Expected all-layer tensor [L, T, D] in {tensor_path}, "
                f"got shape {tuple(tensor.shape)}"
            )

        direct[audio_path] = tensor
        total_bytes += tensor.numel() * tensor.element_size()

    print(f"Loaded {len(direct)} tensors from {index_path}")
    if direct:
        first_key = next(iter(direct))
        first_tensor = direct[first_key]
        print(f"First key: {first_key}")
        print(f"First tensor shape: {tuple(first_tensor.shape)}")
    print(f"Tensor payload size in memory: {total_bytes / (1024 ** 3):.2f} GiB")

    if dry_run:
        print("Dry run requested; not writing output.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(direct, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote direct path -> tensor pickle to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all-layer HuBERT per-utterance shards to a direct pickle."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/speechocean762"),
        help="Directory containing all-layer feature artifacts.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Dataset split to convert.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Per-utterance index pickle. Defaults to *_all_layer_feats.pkl.per_utt.bak.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output pickle. Defaults to *_all_layer_feats_direct.pkl.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow replacing an existing output pickle.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Convert only the first N entries, useful for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate tensors without writing the output pickle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    prefix = _prefix_for_split(args.split)
    index_path = args.index or _default_index_path(data_dir, prefix)
    output_path = args.output or _default_output_path(data_dir, prefix)

    build_direct_pickle(
        data_dir=data_dir,
        index_path=index_path,
        output_path=output_path,
        overwrite_output=args.overwrite_output,
        max_items=args.max_items,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
