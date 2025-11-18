# -*- coding: utf-8 -*-
"""
Standardized dataset module for fluency scoring.

This module provides a unified interface for different datasets:
- All datasets return: (audio_path, labels, features, cluster_indices)
- Labels are always normalized to [0, 1] range (multiply by 0.2)
- Features are pre-extracted HuBERT embeddings
- Cluster indices are optional (for cluster-based models)
"""

import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm


class BaseFluencyDataset(Dataset, ABC):
    """
    Base class for all fluency scoring datasets.

    Standard interface:
        - __getitem__ returns: (audio_path, labels, features, cluster_indices)
        - labels: torch.Tensor of shape (num_aspects,) with values in [0, 1]
        - features: torch.Tensor of shape (seq_len, feature_dim)
        - cluster_indices: torch.Tensor of shape (seq_len,) or None
    """

    def __init__(
        self,
        aspects: List[str],
        kmeans_model: Optional[Any] = None,
        device: str = "cpu",
    ):
        """
        Args:
            aspects: List of aspect names to use (e.g., ["fluency", "prosodic"])
            kmeans_model: Pre-trained kmeans model for clustering (optional)
            device: Device to use for feature extraction
        """
        self.aspects = aspects
        self.kmeans_model = kmeans_model
        self.device = device

        # Standard aspect mapping (0-indexed)
        self.aspect_map = {
            "accuracy": 0,
            "completeness": 1,
            "fluency": 2,
            "prosodic": 3,
            "total": 4,
        }

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...

    @abstractmethod
    def __getitem__(
        self, idx: int
    ) -> Tuple[str, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Get a sample from the dataset.

        Returns:
            audio_path: str, identifier for the audio sample
            labels: torch.Tensor of shape (num_aspects,), normalized to [0, 1]
            features: torch.Tensor of shape (seq_len, feature_dim)
            cluster_indices: torch.Tensor of shape (seq_len,) or None
        """
        ...

    def _normalize_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Normalize labels to [0, 1] range.
        Assumes input labels are in [0, 5] range.
        """
        return labels * 0.2

    def _extract_cluster_indices(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract cluster indices from features using kmeans model.

        Args:
            features: torch.Tensor of shape (seq_len, feature_dim) or (batch, seq_len, feature_dim)

        Returns:
            cluster_indices: torch.Tensor of shape (seq_len,) or (batch, seq_len)
        """
        if self.kmeans_model is None:
            return None

        # Handle both 2D and 3D tensors
        if features.dim() == 2:
            flat_features = features.cpu().numpy()
            cluster_ids_np = self.kmeans_model.predict(flat_features)
            cluster_ids = torch.tensor(cluster_ids_np, dtype=torch.long)
        elif features.dim() == 3:
            B, T, D = features.shape
            flat_features = features.reshape(-1, D).cpu().numpy()
            cluster_ids_np = self.kmeans_model.predict(flat_features)
            cluster_ids = torch.tensor(cluster_ids_np, dtype=torch.long)
            cluster_ids = cluster_ids.reshape(B, T)
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")

        return cluster_ids


class SO762Dataset(BaseFluencyDataset):
    """
    Dataset for SpeechOcean762 data.

    Expects preprocessed data:
        - {data_dir}/{split}/wav.scp: audio file paths
        - data/tr_label_utt.npy or data/te_label_utt.npy: labels
        - data/tr_feats.pkl or data/te_feats.pkl: pre-extracted features
        - data/tr_cluster_index.pkl or data/te_cluster_index.pkl: cluster indices (optional)
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        aspects: List[str],
        kmeans_model: Optional[Any] = None,
        device: str = "cpu",
    ):
        """
        Args:
            data_dir: Root directory of the dataset (e.g., "data/speechocean762")
            split: Dataset split ("train" or "test")
            aspects: List of aspect names to use
            kmeans_model: Pre-trained kmeans model for clustering (optional)
            device: Device to use
        """
        super().__init__(aspects, kmeans_model, device)

        self.data_dir = data_dir
        self.split = split

        # Load audio paths
        wav_scp_path = os.path.join(data_dir, split, "wav.scp")
        paths = np.loadtxt(wav_scp_path, delimiter="\t", dtype=str)
        self.paths = [path.split("\t")[-1] if "\t" in path else path for path in paths]

        # Determine dataset type prefix
        dataset_type = "tr" if split == "train" else "te"

        # Load labels
        label_path = f"data/{dataset_type}_label_utt.npy"
        labels = np.load(label_path)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.labels = self._normalize_labels(self.labels)

        # Load pre-extracted features
        feats_path = f"data/{dataset_type}_feats.pkl"
        with open(feats_path, "rb") as f:
            self.feats = pickle.load(f)

        # Load cluster indices if available
        self.cluster_indices = None
        cluster_path = f"data/{dataset_type}_cluster_index.pkl"
        if os.path.exists(cluster_path):
            with open(cluster_path, "rb") as f:
                self.cluster_indices = pickle.load(f)

        # Extract aspect indices
        self.aspect_indices = [self.aspect_map[aspect] for aspect in aspects]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(
        self, idx: int
    ) -> Tuple[str, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        audio_path = self.paths[idx]

        # Extract labels for requested aspects
        if len(self.aspect_indices) == 1:
            labels = self.labels[idx, self.aspect_indices[0]].unsqueeze(0)
        else:
            labels = self.labels[idx, self.aspect_indices]

        # Get features
        features = self.feats[audio_path]

        # Get cluster indices
        cluster_idx = None
        if self.cluster_indices is not None:
            cluster_idx = self.cluster_indices[audio_path]

        return audio_path, labels, features, cluster_idx


class HuggingFaceDataset(BaseFluencyDataset):
    """
    Dataset for loading from HuggingFace datasets.

    Automatically extracts features and cluster indices during initialization.
    Supports datasets like:
        - eoleedi/ezai-championship2023
        - mispeech/speechocean762
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        aspects: List[str],
        kmeans_model: Optional[Any] = None,
        device: str = "cpu",
        max_duration_sec: float = 30.0,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            dataset_name: HuggingFace dataset identifier (e.g., "eoleedi/ezai-championship2023")
            split: Dataset split ("train" or "test")
            aspects: List of aspect names to use (e.g., ["fluency", "prosodic"])
            kmeans_model: Pre-trained kmeans model for clustering
            device: Device to use for feature extraction
            max_duration_sec: Maximum audio duration in seconds (longer samples will be truncated)
            cache_dir: Directory to cache the dataset
        """
        super().__init__(aspects, kmeans_model, device)

        self.dataset_name = dataset_name
        self.split = split
        self.max_duration_sec = max_duration_sec

        # Load dataset from HuggingFace
        print(f"Loading HuggingFace dataset: {dataset_name}, split: {split}")
        self.dataset = hf_load_dataset(dataset_name, split=split, cache_dir=cache_dir)

        # Load HuBERT feature extractor
        self.feature_extractor = torchaudio.pipelines.HUBERT_LARGE.get_model()
        self.feature_extractor = self.feature_extractor.to(device)
        self.feature_extractor.eval()

        # Pre-extract features and cluster indices
        self._preprocess_dataset()

    def _preprocess_dataset(self):
        """Pre-extract features and cluster indices to speed up training."""
        print(f"Pre-extracting features for {self.split} split...")

        self.feats = []
        self.labels = []
        self.cluster_indices = []
        self.audio_ids = []

        with torch.no_grad():
            for idx, item in enumerate(
                tqdm(self.dataset, desc=f"Processing {self.split}")
            ):
                # Extract audio
                audio = item["audio"]
                array = audio["array"]
                sr = int(audio["sampling_rate"])
                wav = torch.tensor(array, dtype=torch.float32).to(self.device)

                # Ensure mono audio
                if wav.dim() == 2:
                    wav = wav.mean(dim=0)

                # Truncate if too long
                max_samples = int(self.max_duration_sec * sr)
                if wav.shape[0] > max_samples:
                    wav = wav[:max_samples]

                wav_batch = wav.unsqueeze(0)

                # Extract HuBERT features (14th layer)
                audio_embedding, _ = self.feature_extractor.extract_features(wav_batch)
                features = audio_embedding[14]
                if features.dim() == 2:
                    features = features.unsqueeze(0)

                # Extract cluster indices if kmeans model provided
                cluster_idx = None
                if self.kmeans_model is not None:
                    cluster_idx = self._extract_cluster_indices(features)
                    cluster_idx = cluster_idx.squeeze(0)
                    self.cluster_indices.append(cluster_idx.cpu())

                # Store features (move to CPU to save GPU memory)
                self.feats.append(features.squeeze(0).cpu())

                # Extract labels for requested aspects
                labels_list = []
                for aspect in self.aspects:
                    if aspect in item:
                        labels_list.append(item[aspect])
                    else:
                        raise ValueError(
                            f"Aspect '{aspect}' not found in dataset item. Available keys: {list(item.keys())}"
                        )

                labels = torch.tensor(labels_list, dtype=torch.float32)
                labels = self._normalize_labels(labels)
                self.labels.append(labels)

                # Store audio ID
                audio_id = item.get("id", f"sample_{idx}")
                self.audio_ids.append(str(audio_id))

        print(f"Finished pre-extracting {len(self.feats)} samples")

    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(
        self, idx: int
    ) -> Tuple[str, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        audio_id = self.audio_ids[idx]
        labels = self.labels[idx]
        features = self.feats[idx]

        cluster_idx = None
        if len(self.cluster_indices) > 0:
            cluster_idx = self.cluster_indices[idx]

        return audio_id, labels, features, cluster_idx


def custom_collate_fn(batch: List[Tuple]) -> Tuple:
    """
    Custom collate function for batching variable-length sequences.

    Args:
        batch: List of tuples (audio_path, labels, features, cluster_indices)

    Returns:
        paths: List of audio paths
        labels: torch.Tensor of shape (batch_size, num_aspects)
        features: torch.Tensor of shape (batch_size, max_seq_len, feature_dim) - padded
        cluster_indices: torch.Tensor of shape (batch_size, max_seq_len) - padded with -1
    """
    # Sort by feature length (descending) for efficient packing
    batch = sorted(batch, key=lambda x: x[2].shape[0], reverse=True)

    # Extract components
    paths, labels, feats, cluster_idxs = zip(*batch)

    # Stack labels
    labels_tensor = torch.stack(labels)

    # Pad features
    padded_feats = pad_sequence(feats, batch_first=True)

    # Pad cluster indices (use -1 as padding value)
    if cluster_idxs[0] is not None:
        padded_cluster_idxs = pad_sequence(
            cluster_idxs, batch_first=True, padding_value=-1
        )
    else:
        padded_cluster_idxs = None

    return list(paths), labels_tensor, padded_feats, padded_cluster_idxs


def create_dataset(
    dataset_type: str,
    split: str,
    aspects: List[str],
    kmeans_model: Optional[Any] = None,
    device: str = "cpu",
    **kwargs,
) -> BaseFluencyDataset:
    """
    Factory function to create the appropriate dataset.

    Args:
        dataset_type: Type of dataset ("so762", "huggingface", or dataset name)
        split: Dataset split ("train" or "test")
        aspects: List of aspect names to use
        kmeans_model: Pre-trained kmeans model for clustering (optional)
        device: Device to use for feature extraction
        **kwargs: Additional dataset-specific arguments
            For SO762: data_dir (default: "data/speechocean762")
            For HuggingFace: dataset_name, max_duration_sec, cache_dir

    Returns:
        Dataset instance
    """
    if dataset_type.lower() == "so762" or dataset_type.lower() == "speechocean762":
        data_dir = kwargs.get("data_dir", "data/speechocean762")
        return SO762Dataset(
            data_dir=data_dir,
            split=split,
            aspects=aspects,
            kmeans_model=kmeans_model,
            device=device,
        )
    else:
        # Assume it's a HuggingFace dataset name
        return HuggingFaceDataset(
            dataset_name=dataset_type,
            split=split,
            aspects=aspects,
            kmeans_model=kmeans_model,
            device=device,
            max_duration_sec=kwargs.get("max_duration_sec", 30.0),
            cache_dir=kwargs.get("cache_dir", None),
        )
