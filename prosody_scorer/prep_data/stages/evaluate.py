"""Evaluate K-Means clustering quality using on-the-fly HuBERT features."""

from pathlib import Path
from typing import Tuple, Union

import joblib
import torch
import torchaudio
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from .train_kmeans import _extract_feature_batches


def evaluate_clustering(
    dataset_dir: Union[str, Path],
    feat_dir: Union[str, Path],
    model_dir: Union[str, Path],
    feature_batch_size: int = 8,
    device: str = "cuda",
    layer: int = 14,
) -> Tuple[dict, dict]:
    """Evaluate K-Means clustering quality without loading feature pickles."""
    dataset_dir = Path(dataset_dir)
    feat_dir = Path(feat_dir)
    model_dir = Path(model_dir)

    model_path = model_dir / "kmeans_model.joblib"
    print(f"Loading K-means model from: {model_path}")
    cluster = joblib.load(model_path)

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Loading HuBERT-Large model on {device_obj}...")
    hubert = torchaudio.pipelines.HUBERT_LARGE.get_model().to(device_obj)
    hubert.eval()

    print("\n" + "=" * 50)
    print("Evaluating on TRAINING set:")
    print("=" * 50)
    train_metrics = _evaluate_split(
        dataset_dir=dataset_dir,
        feat_dir=feat_dir,
        cluster=cluster,
        model=hubert,
        device=device_obj,
        layer=layer,
        split="train",
        feature_batch_size=feature_batch_size,
    )

    print("\n" + "=" * 50)
    print("Evaluating on TEST set:")
    print("=" * 50)
    test_metrics = _evaluate_split(
        dataset_dir=dataset_dir,
        feat_dir=feat_dir,
        cluster=cluster,
        model=hubert,
        device=device_obj,
        layer=layer,
        split="test",
        feature_batch_size=feature_batch_size,
    )

    print("\n" + "=" * 50)
    print("Evaluation completed!")
    print("=" * 50)

    return train_metrics, test_metrics


def _evaluate_split(
    dataset_dir: Path,
    feat_dir: Path,
    cluster,
    model,
    device: torch.device,
    layer: int,
    split: str,
    feature_batch_size: int,
) -> dict:
    features = torch.cat(
        list(
            _extract_feature_batches(
                dataset_dir=dataset_dir,
                feat_dir=feat_dir,
                split=split,
                model=model,
                device=device,
                layer=layer,
                feature_batch_size=feature_batch_size,
            )
        ),
        dim=0,
    )
    print(f"Feature tensor shape: {features.shape}")

    feat_numpy = features.numpy()
    cluster_labels = cluster.predict(feat_numpy)

    db_score = davies_bouldin_score(feat_numpy, cluster_labels)
    ch_score = calinski_harabasz_score(feat_numpy, cluster_labels)

    print(f"Davies-Bouldin Score: {db_score:.3f} (lower is better)")
    print(f"Calinski-Harabasz Score: {ch_score:.3f} (higher is better)")

    return {
        "davies_bouldin": db_score,
        "calinski_harabasz": ch_score,
        "n_samples": len(feat_numpy),
        "n_clusters": cluster.n_clusters,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate K-Means from audio")
    parser.add_argument("dataset_dir", type=str, help="Dataset directory")
    parser.add_argument("--feat_dir", type=str, default="../data", help="Label directory")
    parser.add_argument("--model_dir", type=str, default="../exp/kmeans", help="Model directory")
    parser.add_argument("--feature_batch_size", type=int, default=8, help="Audio batch size for HuBERT extraction")
    parser.add_argument("--device", type=str, default="cuda", help="Feature extraction device")
    parser.add_argument("--layer", type=int, default=14, help="HuBERT layer")

    args = parser.parse_args()

    train_metrics, test_metrics = evaluate_clustering(
        dataset_dir=args.dataset_dir,
        feat_dir=args.feat_dir,
        model_dir=args.model_dir,
        feature_batch_size=args.feature_batch_size,
        device=args.device,
        layer=args.layer,
    )

    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)
    print(
        f"Train - DB: {train_metrics['davies_bouldin']:.3f}, "
        f"CH: {train_metrics['calinski_harabasz']:.3f}"
    )
    print(
        f"Test  - DB: {test_metrics['davies_bouldin']:.3f}, "
        f"CH: {test_metrics['calinski_harabasz']:.3f}"
    )
