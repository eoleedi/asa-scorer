"""Stage 4: Evaluate K-Means clustering quality."""

import torch
from torch.utils.data import DataLoader
import numpy as np
import joblib
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from pathlib import Path
from typing import Union, Tuple

from ..utils.data_utils import load_wav_scp, AudioDataset, load_pickle


def evaluate_clustering(
    dataset_dir: Union[str, Path],
    feat_dir: Union[str, Path],
    model_dir: Union[str, Path],
) -> Tuple[dict, dict]:
    """
    Evaluate K-Means clustering quality on train and test sets.
    
    Args:
        dataset_dir: Dataset directory containing train/test subdirectories
        feat_dir: Directory containing extracted features
        model_dir: Directory containing trained K-Means model
    
    Returns:
        Tuple of (train_metrics, test_metrics) dictionaries
    """
    dataset_dir = Path(dataset_dir)
    feat_dir = Path(feat_dir)
    model_dir = Path(model_dir)
    
    # Load model
    model_path = model_dir / "kmeans_model.joblib"
    print(f"Loading K-means model from: {model_path}")
    cluster = joblib.load(model_path)
    
    # Evaluate training set
    print("\n" + "="*50)
    print("Evaluating on TRAINING set:")
    print("="*50)
    train_metrics = _evaluate_split(
        dataset_dir=dataset_dir,
        feat_dir=feat_dir,
        cluster=cluster,
        split="train"
    )
    
    # Evaluate test set
    print("\n" + "="*50)
    print("Evaluating on TEST set:")
    print("="*50)
    test_metrics = _evaluate_split(
        dataset_dir=dataset_dir,
        feat_dir=feat_dir,
        cluster=cluster,
        split="test"
    )
    
    print("\n" + "="*50)
    print("Evaluation completed!")
    print("="*50)
    
    return train_metrics, test_metrics


def _evaluate_split(
    dataset_dir: Path,
    feat_dir: Path,
    cluster,
    split: str
) -> dict:
    """Helper function to evaluate a single split."""
    prefix = "tr" if split == "train" else "te"
    
    # Load data
    wav_scp_path = dataset_dir / split / "wav.scp"
    label_file = feat_dir / f"{prefix}_label_utt.npy"
    
    wav_paths = load_wav_scp(wav_scp_path)
    labels = np.load(label_file)
    
    dataset = AudioDataset(wav_paths, labels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load features
    feat_file = feat_dir / f"{prefix}_feats.pkl"
    saved_tensor_dict = load_pickle(feat_file)
    
    # Collect features
    extract_feat_list = []
    for paths, _ in dataloader:
        for path in paths:
            feats = saved_tensor_dict[path]
            extract_feat_list.append(feats)
    
    extract_feat_tensor = torch.concat(extract_feat_list, dim=0)
    print(f"Feature tensor shape: {extract_feat_tensor.shape}")
    
    # Convert to numpy and predict
    feat_numpy = extract_feat_tensor.cpu().numpy()
    cluster_labels = cluster.predict(feat_numpy)
    
    # Calculate metrics
    db_score = davies_bouldin_score(feat_numpy, cluster_labels)
    ch_score = calinski_harabasz_score(feat_numpy, cluster_labels)
    
    print(f"Davies-Bouldin Score: {db_score:.3f} (lower is better ⬇)")
    print(f"Calinski-Harabasz Score: {ch_score:.3f} (higher is better ⬆)")
    
    metrics = {
        "davies_bouldin": db_score,
        "calinski_harabasz": ch_score,
        "n_samples": len(feat_numpy),
        "n_clusters": cluster.n_clusters,
    }
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate K-Means clustering")
    parser.add_argument("dataset_dir", type=str, help="Dataset directory")
    parser.add_argument("--feat_dir", type=str, default="../data", help="Feature directory")
    parser.add_argument("--model_dir", type=str, default="../exp/kmeans", help="Model directory")
    
    args = parser.parse_args()
    
    train_metrics, test_metrics = evaluate_clustering(
        dataset_dir=args.dataset_dir,
        feat_dir=args.feat_dir,
        model_dir=args.model_dir,
    )
    
    print("\n" + "="*50)
    print("Summary:")
    print("="*50)
    print(f"Train - DB: {train_metrics['davies_bouldin']:.3f}, CH: {train_metrics['calinski_harabasz']:.3f}")
    print(f"Test  - DB: {test_metrics['davies_bouldin']:.3f}, CH: {test_metrics['calinski_harabasz']:.3f}")
