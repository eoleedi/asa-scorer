"""Stage 3: Train K-Means clustering model on features."""

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import warnings
import joblib
from pathlib import Path
from typing import Union, Dict
from tqdm import tqdm

from ..utils.data_utils import load_wav_scp, AudioDataset, load_pickle, save_pickle


def train_kmeans_model(
    dataset_dir: Union[str, Path],
    feat_dir: Union[str, Path],
    output_dir: Union[str, Path],
    n_clusters: int = 50,
    max_iter: int = 100,
    batch_size: int = 10000,
    n_init: int = 20,
    random_state: int = 0,
) -> str:
    """
    Train MiniBatch K-Means clustering on training features.
    
    Args:
        dataset_dir: Dataset directory containing train/test subdirectories
        feat_dir: Directory containing extracted features
        output_dir: Directory to save trained model
        n_clusters: Number of clusters (default: 50)
        max_iter: Maximum iterations (default: 100)
        batch_size: Batch size for MiniBatchKMeans (default: 10000)
        n_init: Number of initializations (default: 20)
        random_state: Random seed (default: 0)
    
    Returns:
        Path to saved model file
    """
    dataset_dir = Path(dataset_dir)
    feat_dir = Path(feat_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    wav_scp_path = dataset_dir / "train" / "wav.scp"
    label_file = feat_dir / "tr_label_utt.npy"
    
    print(f"Loading training data...")
    wav_paths = load_wav_scp(wav_scp_path)
    labels = np.load(label_file)
    
    dataset = AudioDataset(wav_paths, labels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load features
    print(f"Loading training features...")
    feat_file = feat_dir / "tr_feats.pkl"
    saved_tensor_dict = load_pickle(feat_file)
    
    # Collect all features
    extract_feat_list = []
    for paths, _ in dataloader:
        for path in paths:
            feats = saved_tensor_dict[path]
            # Flatten to (seq_len, feat_dim) if needed
            if feats.dim() == 3:
                # Shape: (batch=1, seq_len, feat_dim) -> (seq_len, feat_dim)
                feats = feats.squeeze(0)
            extract_feat_list.append(feats.cpu())
    
    # Concatenate all frames from all utterances
    # Each element in extract_feat_list is (seq_len, feat_dim), lengths may vary
    extract_feat_tensor = torch.cat(extract_feat_list, dim=0)
    print(f"Feature tensor shape: {extract_feat_tensor.shape}")
    print(f"Total frames: {extract_feat_tensor.shape[0]}, Feature dimension: {extract_feat_tensor.shape[1]}")
    
    # Suppress sklearn warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)
    
    # Create and train k-means
    print(f"Training MiniBatch K-Means with {n_clusters} clusters...")
    cluster = MiniBatchKMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        batch_size=batch_size,
        n_init=n_init,
        max_no_improvement=100,
        random_state=random_state,
        reassignment_ratio=0.0,
    )
    
    cluster.fit(extract_feat_tensor.numpy())
    print("\033[1;34mK-means training completed!\033[0m")
    
    # Save model
    model_path = output_dir / "kmeans_model.joblib"
    joblib.dump(cluster, model_path)
    print(f"Saved K-means model to: {model_path}")
    
    # Save cluster centers
    cluster_index_dict = {i: cluster.cluster_centers_[i] for i in range(len(cluster.cluster_centers_))}
    centers_path = feat_dir / "cluster_centers.pkl"
    save_pickle(cluster_index_dict, centers_path)
    print(f"Saved cluster centers to: {centers_path}")
    
    # Predict on training set
    print("Generating cluster assignments for training set...")
    _generate_cluster_predictions(dataloader, saved_tensor_dict, cluster, feat_dir, "tr")
    
    # Predict on test set
    print("Generating cluster assignments for test set...")
    wav_scp_path_test = dataset_dir / "test" / "wav.scp"
    label_file_test = feat_dir / "te_label_utt.npy"
    
    wav_paths_test = load_wav_scp(wav_scp_path_test)
    labels_test = np.load(label_file_test)
    
    dataset_test = AudioDataset(wav_paths_test, labels_test)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
    
    feat_file_test = feat_dir / "te_feats.pkl"
    saved_tensor_dict_test = load_pickle(feat_file_test)
    
    _generate_cluster_predictions(dataloader_test, saved_tensor_dict_test, cluster, feat_dir, "te")
    
    print("\033[1;34mCluster prediction completed!\033[0m")
    
    return str(model_path)


def _generate_cluster_predictions(
    dataloader: DataLoader,
    saved_tensor_dict: Dict[str, torch.Tensor],
    cluster,
    feat_dir: Path,
    prefix: str
):
    """Helper function to generate and save cluster predictions."""
    cluster_pred_dict = {}
    
    for paths, _ in dataloader:
        for path in paths:
            feat_tensor = saved_tensor_dict[path]
            # Flatten to (seq_len, feat_dim) if needed
            if feat_tensor.dim() == 3:
                # Shape: (batch=1, seq_len, feat_dim) -> (seq_len, feat_dim)
                feat_tensor = feat_tensor.squeeze(0)
            
            cluster_pred = cluster.predict(feat_tensor.cpu().numpy())
            cluster_pred_tensor = torch.tensor(cluster_pred)
            
            if path not in cluster_pred_dict:
                cluster_pred_dict[path] = cluster_pred_tensor
    
    output_file = feat_dir / f"{prefix}_cluster_index.pkl"
    save_pickle(cluster_pred_dict, output_file)
    print(f"Saved cluster predictions to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train K-Means clustering")
    parser.add_argument("dataset_dir", type=str, help="Dataset directory")
    parser.add_argument("--feat_dir", type=str, default="../data", help="Feature directory")
    parser.add_argument("--output_dir", type=str, default="../exp/kmeans", help="Model output directory")
    parser.add_argument("--n_clusters", type=int, default=50, help="Number of clusters")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum iterations")
    parser.add_argument("--batch_size", type=int, default=10000, help="MiniBatch size")
    parser.add_argument("--n_init", type=int, default=20, help="Number of initializations")
    parser.add_argument("--random_state", type=int, default=0, help="Random seed")
    
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
    )
