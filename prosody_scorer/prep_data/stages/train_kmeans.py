"""Train K-Means on HuBERT features extracted from audio batches."""

from pathlib import Path
from typing import Iterable, Union
import warnings

import joblib
import numpy as np
import torch
import torchaudio
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.data_utils import (
    AudioDataset,
    load_wav_scp,
    resolve_audio_path,
    save_pickle,
)


def _load_audio_batch(paths: Iterable[str], dataset_dir: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    waveforms = []
    lengths = []

    for path in paths:
        audio_path = resolve_audio_path(path, dataset_dir, split)
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        waveform = waveform.to(dtype=torch.float32)
        waveforms.append(waveform)
        lengths.append(waveform.numel())

    length_tensor = torch.tensor(lengths, dtype=torch.long)
    padded = waveforms[0].new_zeros(len(waveforms), int(length_tensor.max().item()))
    for i, waveform in enumerate(waveforms):
        padded[i, : waveform.numel()] = waveform

    return padded, length_tensor


def _extract_feature_batches(
    dataset_dir: Path,
    feat_dir: Path,
    split: str,
    model,
    device: torch.device,
    layer: int,
    feature_batch_size: int,
):
    prefix = "tr" if split == "train" else "te"
    wav_paths = load_wav_scp(dataset_dir / split / "wav.scp")
    labels = np.load(feat_dir / f"{prefix}_label_utt.npy")
    dataloader = DataLoader(
        AudioDataset(wav_paths, labels),
        batch_size=feature_batch_size,
        shuffle=False,
    )

    for paths, _ in tqdm(dataloader, desc=f"Extracting {split} HuBERT batches"):
        audio, lengths = _load_audio_batch(paths, dataset_dir, split)
        with torch.inference_mode():
            embeddings, feature_lengths = model.extract_features(
                audio.to(device), lengths=lengths.to(device)
            )
        features = embeddings[layer].detach().cpu()
        if feature_lengths is None:
            feature_lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long
            )
        feature_lengths = feature_lengths.detach().cpu().to(dtype=torch.long)

        for i, feature_length in enumerate(feature_lengths.tolist()):
            yield features[i, :feature_length]


def train_kmeans_model(
    dataset_dir: Union[str, Path],
    feat_dir: Union[str, Path],
    output_dir: Union[str, Path],
    n_clusters: int = 50,
    max_iter: int = 100,
    batch_size: int = 10000,
    n_init: int = 20,
    random_state: int = 0,
    feature_batch_size: int = 8,
    device: str = "cuda",
    layer: int = 14,
) -> str:
    """
    Train MiniBatch K-Means on HuBERT features extracted on the fly.

    No feature pickle or cluster-index pickle is written; training computes
    cluster indices at batch time from the saved K-Means model.
    """
    dataset_dir = Path(dataset_dir)
    feat_dir = Path(feat_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Loading HuBERT-Large model on {device_obj}...")
    model = torchaudio.pipelines.HUBERT_LARGE.get_model().to(device_obj)
    model.eval()

    warnings.simplefilter(action="ignore", category=FutureWarning)
    cluster = MiniBatchKMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        batch_size=batch_size,
        n_init=n_init,
        max_no_improvement=100,
        random_state=random_state,
        reassignment_ratio=0.0,
    )

    print(f"Training MiniBatch K-Means with {n_clusters} clusters from audio...")
    frame_buffer = []
    total_frames = 0
    min_partial_fit_frames = max(batch_size, n_clusters)
    for features in _extract_feature_batches(
        dataset_dir=dataset_dir,
        feat_dir=feat_dir,
        split="train",
        model=model,
        device=device_obj,
        layer=layer,
        feature_batch_size=feature_batch_size,
    ):
        frame_buffer.append(features)
        buffered_frames = sum(tensor.shape[0] for tensor in frame_buffer)
        if buffered_frames >= min_partial_fit_frames:
            batch = torch.cat(frame_buffer, dim=0)
            cluster.partial_fit(batch.numpy())
            total_frames += batch.shape[0]
            frame_buffer.clear()

    if frame_buffer:
        batch = torch.cat(frame_buffer, dim=0)
        cluster.partial_fit(batch.numpy())
        total_frames += batch.shape[0]

    print(f"\033[1;34mK-means training completed on {total_frames} frames!\033[0m")

    model_path = output_dir / "kmeans_model.joblib"
    joblib.dump(cluster, model_path)
    print(f"Saved K-means model to: {model_path}")

    cluster_index_dict = {
        i: cluster.cluster_centers_[i] for i in range(len(cluster.cluster_centers_))
    }
    centers_path = feat_dir / "cluster_centers.pkl"
    save_pickle(cluster_index_dict, centers_path)
    print(f"Saved cluster centers to: {centers_path}")

    return str(model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train K-Means from audio")
    parser.add_argument("dataset_dir", type=str, help="Dataset directory")
    parser.add_argument("--feat_dir", type=str, default="../data", help="Label directory")
    parser.add_argument("--output_dir", type=str, default="../exp/kmeans", help="Model output directory")
    parser.add_argument("--n_clusters", type=int, default=50, help="Number of clusters")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum iterations")
    parser.add_argument("--batch_size", type=int, default=10000, help="MiniBatch K-Means frame batch size")
    parser.add_argument("--feature_batch_size", type=int, default=8, help="Audio batch size for HuBERT extraction")
    parser.add_argument("--n_init", type=int, default=20, help="Number of initializations")
    parser.add_argument("--random_state", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Feature extraction device")
    parser.add_argument("--layer", type=int, default=14, help="HuBERT layer")

    args = parser.parse_args()

    train_kmeans_model(
        dataset_dir=args.dataset_dir,
        feat_dir=args.feat_dir,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        n_init=args.n_init,
        random_state=args.random_state,
        feature_batch_size=args.feature_batch_size,
        device=args.device,
        layer=args.layer,
    )
