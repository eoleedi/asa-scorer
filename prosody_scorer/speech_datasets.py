# -*- coding: utf-8 -*-
"""
Standardized dataset module for scoring.

This module provides a unified interface for different datasets:
- All datasets return: (audio_path, labels, features, cluster_indices)
- Labels are always normalized to [0, 1] range (multiply by 0.2)
- Features are pre-extracted HuBERT embeddings
- Cluster indices are optional (for cluster-based models)
"""

import os
import pickle
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm


class BaseDataset(Dataset, ABC):
    """
    Base class for all scoring datasets.

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


class SO762Dataset(BaseDataset):
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
        feature_type: str = "ssl",
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
        self.feature_type = feature_type.lower()

        # Determine dataset type prefix
        dataset_type = "tr" if split == "train" else "te"

        # Load labels
        label_path = os.path.join(data_dir, f"{dataset_type}_label_utt.npy")
        if not os.path.exists(label_path):
            # Fallback to old path for backward compatibility or if data is in root data/
            label_path = f"data/{dataset_type}_label_utt.npy"

        labels = np.load(label_path)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.labels = self._normalize_labels(self.labels)

        # Load pre-extracted features
        if self.feature_type == "handcrafted":
            feats_filename = f"{dataset_type}_handcrafted_feats.pkl"
        elif self.feature_type == "fdmpa":
            feats_filename = f"{dataset_type}_feats.pkl"
        else:
            feats_filename = f"{dataset_type}_feats.pkl"

        feats_path = os.path.join(data_dir, feats_filename)
        if not os.path.exists(feats_path):
            feats_path = f"data/{feats_filename}"

        with open(feats_path, "rb") as f:
            self.feats = pickle.load(f)

        self.hc_feats = None
        if self.feature_type == "fdmpa":
            hc_feats_filename = f"{dataset_type}_handcrafted_feats.pkl"
            hc_feats_path = os.path.join(data_dir, hc_feats_filename)
            if not os.path.exists(hc_feats_path):
                hc_feats_path = f"data/{hc_feats_filename}"
            with open(hc_feats_path, "rb") as f:
                self.hc_feats = pickle.load(f)

        # Load audio paths
        wav_scp_path = os.path.join(data_dir, split, "wav.scp")
        if not os.path.exists(wav_scp_path):
            nested_wav_scp_path = os.path.join(data_dir, "so762", split, "wav.scp")
            if os.path.exists(nested_wav_scp_path):
                wav_scp_path = nested_wav_scp_path
        if os.path.exists(wav_scp_path):
            self.paths = []
            with open(wav_scp_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    self.paths.append(parts[1])
        else:
            # If wav.scp doesn't exist, use keys from feats
            print(
                f"Warning: {wav_scp_path} not found. Using keys from features dictionary as paths."
            )
            # Do NOT sort keys, as dictionary insertion order (Python 3.7+) likely preserves
            # the order from the original wav.scp used to generate the features and labels.
            self.paths = list(self.feats.keys())

        # Load cluster indices if available
        self.cluster_indices = None
        cluster_path = os.path.join(data_dir, f"{dataset_type}_cluster_index.pkl")
        if not os.path.exists(cluster_path):
            cluster_path = f"data/{dataset_type}_cluster_index.pkl"

        if os.path.exists(cluster_path):
            with open(cluster_path, "rb") as f:
                self.cluster_indices = pickle.load(f)

        if self.feature_type == "handcrafted" or self.feature_type == "fdmpa":
            self.cluster_indices = None

        # Extract aspect indices
        self.aspect_indices = [self.aspect_map[aspect] for aspect in aspects]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple:
        audio_path = self.paths[idx]

        # Extract labels for requested aspects
        if len(self.aspect_indices) == 1:
            labels = self.labels[idx, self.aspect_indices[0]].unsqueeze(0)
        else:
            labels = self.labels[idx, self.aspect_indices]

        features = self.feats[audio_path]
        if features.dim() == 3:
            features = features.squeeze(0)

        hc_features = None
        if self.feature_type == "fdmpa":
            hc_features = self.hc_feats[audio_path]
            if hc_features.dim() == 3:
                hc_features = hc_features.squeeze(0)

        # Get cluster indices
        cluster_idx = None
        if self.cluster_indices is not None:
            cluster_idx = self.cluster_indices[audio_path]

        if self.feature_type == "fdmpa":
            return audio_path, labels, features, hc_features, cluster_idx
        return audio_path, labels, features, cluster_idx


class HuggingFaceDataset(BaseDataset):
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
        feature_type: str = "ssl",
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
            feature_type: Feature source to use ("ssl", "handcrafted", "fdmpa")
        """
        super().__init__(aspects, kmeans_model, device)

        self.dataset_name = dataset_name
        self.split = split
        self.max_duration_sec = max_duration_sec
        self.feature_type = feature_type.lower()
        self.cache_dir = cache_dir

        # Load dataset from HuggingFace
        print(f"Loading HuggingFace dataset: {dataset_name}, split: {split}")
        self.dataset = hf_load_dataset(dataset_name, split=split, cache_dir=cache_dir)

        # Load HuBERT feature extractor
        self.feature_extractor = torchaudio.pipelines.HUBERT_LARGE.get_model()
        self.feature_extractor = self.feature_extractor.to(device)
        self.feature_extractor.eval()

        # Pre-extract features and cluster indices
        self._preprocess_dataset()

    def _get_feature_cache_paths(self):
        """Return cache file paths for pre-extracted features."""
        dataset_leaf = self.dataset_name.split("/")[-1]
        safe_dataset = re.sub(r"[^a-zA-Z0-9_.-]", "_", dataset_leaf)

        if self.cache_dir:
            base_dir = os.path.join(
                self.cache_dir, "prosody_feature_cache", safe_dataset
            )
        else:
            # Default local cache (keeps behavior close to SO762 pre-extracted feature files)
            base_dir = os.path.join("data", safe_dataset)

        os.makedirs(base_dir, exist_ok=True)

        prefix = "tr" if self.split == "train" else "te"
        ssl_cache_path = os.path.join(base_dir, f"{prefix}_feats.pkl")
        hc_cache_path = os.path.join(base_dir, f"{prefix}_handcrafted_feats.pkl")
        return ssl_cache_path, hc_cache_path

    def _extract_handcrafted_features(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract handcrafted features (loudness/pitch/periodicity/ppg) from waveform."""
        import promonet

        wav_np = wav.detach().cpu().numpy()
        handcrafted_dict = promonet.preprocess.from_audio(
            wav_np,
            sample_rate=sr,
            features=["loudness", "pitch", "periodicity", "ppg"],
        )

        feature_list = []
        for feat_name in ["loudness", "pitch", "periodicity", "ppg"]:
            feat_data = handcrafted_dict[feat_name]
            if isinstance(feat_data, np.ndarray):
                feat_data = torch.tensor(feat_data, dtype=torch.float32)
            else:
                feat_data = feat_data.to(dtype=torch.float32)

            if feat_data.ndim == 1:
                feat_data = feat_data.unsqueeze(-1)
            feature_list.append(feat_data)

        return torch.cat(feature_list, dim=-1)

    def _preprocess_dataset(self):
        """Pre-extract features and cluster indices to speed up training."""
        print(f"Pre-extracting features for {self.split} split...")

        ssl_cache_path, hc_cache_path = self._get_feature_cache_paths()

        self.feats = []
        self.hc_feats = []
        self.labels = []
        self.cluster_indices = []
        self.audio_ids = []

        cached_ssl = None
        cached_hc = None
        if ssl_cache_path and os.path.exists(ssl_cache_path):
            with open(ssl_cache_path, "rb") as f:
                cached_ssl = pickle.load(f)
            print(f"Loaded cached SSL features: {ssl_cache_path}")

        if self.feature_type in ["handcrafted", "fdmpa"]:
            if hc_cache_path and os.path.exists(hc_cache_path):
                with open(hc_cache_path, "rb") as f:
                    cached_hc = pickle.load(f)
                print(f"Loaded cached handcrafted features: {hc_cache_path}")

        with torch.no_grad():
            for idx, item in enumerate(
                tqdm(self.dataset, desc=f"Processing {self.split}")
            ):
                audio_id = str(item.get("id", f"sample_{idx}"))

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

                if cached_ssl is not None and audio_id in cached_ssl:
                    features = cached_ssl[audio_id]
                    if features.dim() == 2:
                        features = features.unsqueeze(0)
                else:
                    wav_batch = wav.unsqueeze(0)

                    # Extract HuBERT features (14th layer)
                    audio_embedding, _ = self.feature_extractor.extract_features(
                        wav_batch
                    )
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

                # Store handcrafted features when requested
                if self.feature_type in ["handcrafted", "fdmpa"]:
                    if cached_hc is not None and audio_id in cached_hc:
                        hc_features = cached_hc[audio_id]
                    else:
                        hc_features = self._extract_handcrafted_features(wav, sr)
                    self.hc_feats.append(hc_features.cpu())

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
                self.audio_ids.append(str(audio_id))

        if ssl_cache_path and cached_ssl is None:
            ssl_dict = {aid: feat for aid, feat in zip(self.audio_ids, self.feats)}
            with open(ssl_cache_path, "wb") as f:
                pickle.dump(ssl_dict, f)
            print(f"Saved cached SSL features: {ssl_cache_path}")

        if (
            self.feature_type in ["handcrafted", "fdmpa"]
            and hc_cache_path
            and cached_hc is None
        ):
            hc_dict = {aid: feat for aid, feat in zip(self.audio_ids, self.hc_feats)}
            with open(hc_cache_path, "wb") as f:
                pickle.dump(hc_dict, f)
            print(f"Saved cached handcrafted features: {hc_cache_path}")

        print(f"Finished pre-extracting {len(self.feats)} samples")

    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(
        self, idx: int
    ) -> Tuple[str, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        audio_id = self.audio_ids[idx]
        labels = self.labels[idx]
        features = self.feats[idx]
        hc_features = None
        if self.feature_type in ["handcrafted", "fdmpa"]:
            hc_features = self.hc_feats[idx]

        cluster_idx = None
        if (
            self.feature_type not in ["handcrafted", "fdmpa"]
            and len(self.cluster_indices) > 0
        ):
            cluster_idx = self.cluster_indices[idx]

        if self.feature_type == "fdmpa":
            return audio_id, labels, features, hc_features, cluster_idx
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

    return (
        list(paths),
        labels_tensor,
        padded_feats,
        padded_cluster_idxs,
    )


def hcssl_collate_fn(batch: List[Tuple]) -> Tuple:
    """
    Collate function for CrossAttnHCSSLScorer.
    Expects batch items with 5 elements: (path, labels, ssl_feats, hc_feats, cluster_idx)
    Returns: (paths, labels, ssl_feats, hc_feats, None) where None replaces cluster_idx
    This matches the 5-element format expected by unpack_batch().
    """
    batch = sorted(batch, key=lambda x: x[2].shape[0], reverse=True)
    paths, labels, ssl_feats, hc_feats, cluster_idxs = zip(*batch)

    labels_tensor = torch.stack(labels)
    padded_ssl_feats = pad_sequence(ssl_feats, batch_first=True)
    padded_hc_feats = pad_sequence(hc_feats, batch_first=True)

    return (
        list(paths),
        labels_tensor,
        padded_ssl_feats,
        padded_hc_feats,
        None,  # No cluster indices for CrossAttnHCSSLScorer
    )


def fdmpa_collate_fn(batch: List[Tuple]) -> Tuple:
    batch = sorted(batch, key=lambda x: x[2].shape[0], reverse=True)
    paths, labels, ssl_feats, hc_feats, cluster_idxs = zip(*batch)

    labels_tensor = torch.stack(labels)
    padded_ssl_feats = pad_sequence(ssl_feats, batch_first=True)
    padded_hc_feats = pad_sequence(hc_feats, batch_first=True)

    if cluster_idxs[0] is not None:
        padded_cluster_idxs = pad_sequence(
            cluster_idxs, batch_first=True, padding_value=-1
        )
    else:
        padded_cluster_idxs = None

    return (
        list(paths),
        labels_tensor,
        padded_ssl_feats,
        padded_hc_feats,
        padded_cluster_idxs,
    )


def create_dataset(
    dataset_type: str,
    split: str,
    aspects: List[str],
    kmeans_model: Optional[Any] = None,
    device: str = "cpu",
    **kwargs,
) -> BaseDataset:
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
            feature_type=kwargs.get("feature_type", "ssl"),
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
            feature_type=kwargs.get("feature_type", "ssl"),
        )
