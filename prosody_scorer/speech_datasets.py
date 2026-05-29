# -*- coding: utf-8 -*-
"""
Standardized dataset module for scoring.

This module provides a unified interface for different datasets:
- Datasets return raw audio; collators extract and pad features per batch
- Labels are always normalized to [0, 1] range (multiply by 0.2)
- Cluster indices are optional (for cluster-based models)
"""

import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset as hf_load_dataset


def resample_sequence_to_length(sequence: torch.Tensor, target_len: int) -> torch.Tensor:
    """Resample a [T, D] sequence to target_len frames along time."""
    if sequence.dim() != 2:
        raise ValueError(
            f"Expected sequence with shape [T, D], got {tuple(sequence.shape)}"
        )
    if target_len < 0:
        raise ValueError(f"target_len must be non-negative, got {target_len}")
    if sequence.shape[0] == target_len:
        return sequence
    if target_len == 0:
        return sequence.new_zeros(0, sequence.shape[-1])
    if sequence.shape[0] == 0:
        return sequence.new_zeros(target_len, sequence.shape[-1])

    x = sequence.transpose(0, 1).unsqueeze(0)
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return x.squeeze(0).transpose(0, 1)


class BaseDataset(Dataset, ABC):
    """
    Base class for all scoring datasets.

    Standard interface:
        - __getitem__ returns: (audio_path, labels, waveform, sample_rate)
        - labels: torch.Tensor of shape (num_aspects,) with values in [0, 1]
        - waveform: torch.Tensor of shape (num_samples,)
        - sample_rate: int
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
    ) -> Tuple[str, torch.Tensor, torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Returns:
            audio_path: str, identifier for the audio sample
            labels: torch.Tensor of shape (num_aspects,), normalized to [0, 1]
            waveform: torch.Tensor of shape (num_samples,)
            sample_rate: int
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
    """SpeechOcean762 dataset that returns raw waveforms and labels."""

    def __init__(
        self,
        data_dir: str,
        split: str,
        aspects: List[str],
        kmeans_model: Optional[Any] = None,
        device: str = "cpu",
        feature_type: str = "ssl",
        on_the_fly_features: bool = True,
    ):
        super().__init__(aspects, kmeans_model, device)
        self.data_dir = data_dir
        self.split = split
        self.feature_type = feature_type.lower()

        dataset_type = "tr" if split == "train" else "te"
        label_path = os.path.join(data_dir, f"{dataset_type}_label_utt.npy")
        if not os.path.exists(label_path):
            label_path = f"data/{dataset_type}_label_utt.npy"
        labels = np.load(label_path)
        self.labels = self._normalize_labels(torch.tensor(labels, dtype=torch.float32))

        wav_scp_path = os.path.join(data_dir, split, "wav.scp")
        if not os.path.exists(wav_scp_path):
            nested_wav_scp_path = os.path.join(data_dir, "so762", split, "wav.scp")
            if os.path.exists(nested_wav_scp_path):
                wav_scp_path = nested_wav_scp_path
        if not os.path.exists(wav_scp_path):
            raise FileNotFoundError(
                f"{wav_scp_path} is required for on-the-fly feature extraction."
            )
        self.wav_scp_dir = os.path.dirname(wav_scp_path)

        self.paths = []
        with open(wav_scp_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.paths.append(parts[1])

        self.aspect_indices = [self.aspect_map[aspect] for aspect in aspects]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple:
        audio_path = self.paths[idx]
        if len(self.aspect_indices) == 1:
            labels = self.labels[idx, self.aspect_indices[0]].unsqueeze(0)
        else:
            labels = self.labels[idx, self.aspect_indices]

        resolved_path = audio_path
        if not os.path.exists(resolved_path):
            candidates = [
                os.path.join(self.wav_scp_dir, audio_path),
                os.path.join(self.data_dir, self.split, audio_path),
                os.path.join(self.data_dir, "so762", self.split, audio_path),
                os.path.join(self.data_dir, audio_path),
            ]
            resolved_path = next(
                (candidate for candidate in candidates if os.path.exists(candidate)),
                audio_path,
            )
        wav, sr = torchaudio.load(resolved_path)
        if wav.dim() == 2:
            wav = wav.mean(dim=0)
        return audio_path, labels, wav.to(dtype=torch.float32), int(sr)


class HuggingFaceDataset(BaseDataset):
    """HuggingFace dataset that returns raw waveforms and labels."""

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
        on_the_fly_features: bool = True,
    ):
        super().__init__(aspects, kmeans_model, device)
        self.dataset_name = dataset_name
        self.split = split
        self.max_duration_sec = max_duration_sec
        self.feature_type = feature_type.lower()
        self.cache_dir = cache_dir

        print(f"Loading HuggingFace dataset: {dataset_name}, split: {split}")
        self.dataset = hf_load_dataset(dataset_name, split=split, cache_dir=cache_dir)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple:
        item = self.dataset[idx]
        audio_id = str(item.get("id", f"sample_{idx}"))
        audio = item["audio"]
        wav = torch.tensor(audio["array"], dtype=torch.float32)
        sr = int(audio["sampling_rate"])
        if wav.dim() == 2:
            wav = wav.mean(dim=0)
        max_samples = int(self.max_duration_sec * sr)
        if wav.shape[0] > max_samples:
            wav = wav[:max_samples]

        labels_list = []
        for aspect in self.aspects:
            if aspect not in item:
                raise ValueError(
                    f"Aspect '{aspect}' not found in dataset item. Available keys: {list(item.keys())}"
                )
            labels_list.append(item[aspect])
        labels = self._normalize_labels(torch.tensor(labels_list, dtype=torch.float32))
        return audio_id, labels, wav, sr


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
    hc_feats = [
        resample_sequence_to_length(hc_feat, ssl_feat.shape[0])
        for ssl_feat, hc_feat in zip(ssl_feats, hc_feats)
    ]
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
    hc_feats = [
        resample_sequence_to_length(hc_feat, ssl_feat.shape[0])
        for ssl_feat, hc_feat in zip(ssl_feats, hc_feats)
    ]
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


def all_layer_fdmpa_collate_fn(batch: List[Tuple]) -> Tuple:
    batch = sorted(batch, key=lambda x: x[2].shape[-2], reverse=True)
    paths, labels, ssl_layers, hc_feats, cluster_idxs = zip(*batch)

    labels_tensor = torch.stack(labels)
    hc_feats = [
        resample_sequence_to_length(hc_feat, ssl_layer.shape[1])
        for ssl_layer, hc_feat in zip(ssl_layers, hc_feats)
    ]
    num_layers = ssl_layers[0].shape[0]
    feat_dim = ssl_layers[0].shape[-1]
    max_ssl_len = max(feat.shape[-2] for feat in ssl_layers)
    padded_ssl_layers = ssl_layers[0].new_zeros(
        len(ssl_layers), num_layers, max_ssl_len, feat_dim
    )
    for i, feat in enumerate(ssl_layers):
        if feat.dim() != 3:
            raise ValueError(
                f"Expected all-layer SSL feature [L, T, D], got {tuple(feat.shape)}"
            )
        if feat.shape[0] != num_layers or feat.shape[-1] != feat_dim:
            raise ValueError(
                "All all-layer SSL features in a batch must share layer count and feature dimension."
            )
        padded_ssl_layers[i, :, : feat.shape[1], :] = feat

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
        padded_ssl_layers,
        padded_hc_feats,
        padded_cluster_idxs,
    )


class OnTheFlyFeatureCollator:
    """Extract audio features for each batch, padding waveforms to the batch maximum."""

    def __init__(
        self,
        feature_type: str = "ssl",
        device: str = "cpu",
        kmeans_model: Optional[Any] = None,
        ssl_layer: int = 14,
        sample_rate: int = 16000,
        timing_enabled: bool = False,
        timing_report_every: int = 20,
    ):
        self.feature_type = feature_type.lower()
        self.device = torch.device(device)
        self.kmeans_model = kmeans_model
        self.ssl_layer = ssl_layer
        self.sample_rate = sample_rate
        self.feature_extractor = None
        self.timing_enabled = timing_enabled
        self.timing_report_every = max(1, int(timing_report_every))
        self._timing_sums_ms = {
            "ssl_extract": 0.0,
            "handcrafted_extract": 0.0,
            "resample": 0.0,
            "pad": 0.0,
        }
        self._hc_stage_sums_ms = {
            "hc_prepare": 0.0,
            "hc_loudness": 0.0,
            "hc_penn": 0.0,
            "hc_ppg": 0.0,
            "hc_grid_sample": 0.0,
            "hc_postprocess": 0.0,
        }
        self._timing_batch_count = 0

    def _time_call(self, stage: str, fn):
        if not self.timing_enabled:
            return fn()
        t0 = time.perf_counter()
        out = fn()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if stage in self._timing_sums_ms:
            self._timing_sums_ms[stage] += elapsed_ms
        return out

    def _report_timing_if_needed(self) -> None:
        if not self.timing_enabled:
            return
        self._timing_batch_count += 1
        if self._timing_batch_count % self.timing_report_every != 0:
            return

        denom = float(self.timing_report_every)
        avg_ssl = self._timing_sums_ms["ssl_extract"] / denom
        avg_hc = self._timing_sums_ms["handcrafted_extract"] / denom
        avg_resample = self._timing_sums_ms["resample"] / denom
        avg_pad = self._timing_sums_ms["pad"] / denom
        total = avg_ssl + avg_hc + avg_resample + avg_pad

        print(
            "[COLLATE_TIMING] "
            f"batches={self._timing_batch_count} "
            f"avg_ssl_extract_ms={avg_ssl:.2f} "
            f"avg_handcrafted_extract_ms={avg_hc:.2f} "
            f"avg_resample_ms={avg_resample:.2f} "
            f"avg_pad_ms={avg_pad:.2f} "
            f"avg_total_ms={total:.2f}"
        )

        if avg_hc > 0.0:
            hc_stage_avg = {
                k: v / denom for k, v in self._hc_stage_sums_ms.items()
            }
            ordered = sorted(hc_stage_avg.items(), key=lambda x: x[1], reverse=True)
            top = " ".join(
                f"{name}={value:.2f}ms({(100.0 * value / max(1e-9, avg_hc)):.1f}%)"
                for name, value in ordered
            )
            print(f"[HC_TIMING_BREAKDOWN] avg_per_batch {top}")

        for key in self._timing_sums_ms:
            self._timing_sums_ms[key] = 0.0
        for key in self._hc_stage_sums_ms:
            self._hc_stage_sums_ms[key] = 0.0

    def _get_feature_extractor(self):
        if self.feature_extractor is None:
            self.feature_extractor = torchaudio.pipelines.HUBERT_LARGE.get_model()
            self.feature_extractor = self.feature_extractor.to(self.device)
            self.feature_extractor.eval()
        return self.feature_extractor

    def _prepare_waveform(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        wav = wav.to(dtype=torch.float32)
        if wav.dim() == 2:
            wav = wav.mean(dim=0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav

    def _extract_handcrafted_features(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        import math
        import penn
        import ppgs
        import promonet

        def _stage_call(stage: str, fn):
            if not self.timing_enabled:
                return fn()
            t0 = time.perf_counter()
            out = fn()
            self._hc_stage_sums_ms[stage] += (time.perf_counter() - t0) * 1000.0
            return out

        wav = _stage_call(
            "hc_prepare", lambda: self._prepare_waveform(wav, sr).detach().cpu().unsqueeze(0)
        )

        # Prefer running penn decoding on GPU when available and requested
        penn_gpu = None
        try:
            # Honor penn's module-level default if set by training/test script
            import penn as _penn
            penn_gpu = getattr(_penn, '_DEFAULT_GPU', None)
            if penn_gpu is None:
                if torch.cuda.is_available() and getattr(self, 'device', None) is not None and self.device.type == 'cuda':
                    penn_gpu = self.device.index if self.device.index is not None else 0
        except Exception:
            penn_gpu = None
        loudness = _stage_call(
            "hc_loudness",
            lambda: promonet.preprocess.loudness.from_audio(wav, promonet.LOUDNESS_BANDS),
        )
        pitch, periodicity = _stage_call(
            "hc_penn",
            lambda: penn.from_audio(
                wav,
                sample_rate=self.sample_rate,
                hopsize=promonet.convert.samples_to_seconds(promonet.HOPSIZE),
                fmin=promonet.FMIN,
                fmax=promonet.FMAX,
                batch_size=2048,
                center="half-hop",
                decoder="viterbi" if promonet.VITERBI_DECODE_PITCH else "argmax",
                interp_unvoiced_at=None
                if promonet.VITERBI_DECODE_PITCH
                else promonet.VOICING_THRESHOLD,
                gpu=penn_gpu,
            ),
        )
        ppg = _stage_call("hc_ppg", lambda: ppgs.from_audio(wav, self.sample_rate, gpu=None))

        # Work around promonet 0.x passing an integer to torchaudio.resample
        # when aligning PPGs to the promonet frame grid.
        def _grid_and_softmax():
            resampled_samples = math.ceil(
                wav.shape[-1] * promonet.SAMPLE_RATE / self.sample_rate
            )
            target_length = promonet.convert.samples_to_frames(resampled_samples)
            p = promonet.edit.grid.sample(
                ppg,
                promonet.edit.grid.of_length(ppg, target_length),
                promonet.PPG_INTERP_METHOD,
            )
            return torch.softmax(torch.log(p + 1e-8), -2)

        ppg = _stage_call("hc_grid_sample", _grid_and_softmax)

        handcrafted_dict = {
            "loudness": loudness,
            "pitch": pitch,
            "periodicity": periodicity,
            "ppg": ppg,
        }

        def _postprocess():
            feature_list = []
            for feat_name in ["loudness", "pitch", "periodicity", "ppg"]:
                feat_data = handcrafted_dict[feat_name]
                if isinstance(feat_data, np.ndarray):
                    feat_data = torch.tensor(feat_data, dtype=torch.float32)
                else:
                    feat_data = feat_data.detach().cpu().to(dtype=torch.float32)
                feat_data = feat_data.squeeze(0)
                if feat_data.ndim == 1:
                    feat_data = feat_data.unsqueeze(-1)
                elif feat_data.shape[0] < feat_data.shape[-1]:
                    feat_data = feat_data.transpose(0, 1)
                feature_list.append(feat_data)

            min_len = min(feature.shape[0] for feature in feature_list)
            feature_list = [feature[:min_len] for feature in feature_list]
            return torch.cat(feature_list, dim=-1)

        return _stage_call("hc_postprocess", _postprocess)

    def _extract_ssl_features(
        self, wavs: Tuple[torch.Tensor, ...], sample_rates: Tuple[int, ...]
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        prepared = [
            self._prepare_waveform(wav, int(sr)) for wav, sr in zip(wavs, sample_rates)
        ]
        lengths = torch.tensor([wav.numel() for wav in prepared], dtype=torch.long)
        max_len = int(lengths.max().item())
        padded = prepared[0].new_zeros(len(prepared), max_len)
        for i, wav in enumerate(prepared):
            padded[i, : wav.numel()] = wav

        model = self._get_feature_extractor()
        with torch.inference_mode():
            embeddings, feature_lengths = model.extract_features(
                padded.to(self.device), lengths=lengths.to(self.device)
            )

        if feature_lengths is None:
            feature_lengths = torch.full(
                (len(prepared),), embeddings[0].shape[1], dtype=torch.long
            )
        feature_lengths = feature_lengths.detach().cpu().to(dtype=torch.long)

        if self.feature_type == "all_layers_fdmpa":
            layer_tensor = torch.stack(embeddings, dim=1).detach().cpu()
            ssl_features = [
                layer_tensor[i, :, : int(feature_lengths[i].item()), :]
                for i in range(layer_tensor.shape[0])
            ]
        else:
            selected = embeddings[self.ssl_layer].detach().cpu()
            ssl_features = [
                selected[i, : int(feature_lengths[i].item()), :]
                for i in range(selected.shape[0])
            ]

        cluster_indices = None
        if self.kmeans_model is not None and self.feature_type == "ssl":
            cluster_indices = []
            for features in ssl_features:
                cluster_np = self.kmeans_model.predict(features.numpy())
                cluster_indices.append(torch.tensor(cluster_np, dtype=torch.long))

        return ssl_features, cluster_indices

    def __call__(self, batch: List[Tuple]) -> Tuple:
        batch = sorted(batch, key=lambda x: x[2].shape[0], reverse=True)
        paths, labels, wavs, sample_rates = zip(*batch)
        labels_tensor = torch.stack(labels)

        if self.feature_type == "handcrafted":
            hc_feats = self._time_call(
                "handcrafted_extract",
                lambda: [
                    self._extract_handcrafted_features(wav, int(sr))
                    for wav, sr in zip(wavs, sample_rates)
                ],
            )
            padded_hc = self._time_call(
                "pad", lambda: pad_sequence(hc_feats, batch_first=True)
            )
            self._report_timing_if_needed()
            return list(paths), labels_tensor, padded_hc, None

        ssl_feats, cluster_idxs = self._time_call(
            "ssl_extract", lambda: self._extract_ssl_features(wavs, sample_rates)
        )

        if self.feature_type == "all_layers_fdmpa":
            num_layers = ssl_feats[0].shape[0]
            feat_dim = ssl_feats[0].shape[-1]
            max_ssl_len = max(feat.shape[1] for feat in ssl_feats)
            def _pad_all_layers():
                padded = ssl_feats[0].new_zeros(
                    len(ssl_feats), num_layers, max_ssl_len, feat_dim
                )
                for i, feat in enumerate(ssl_feats):
                    padded[i, :, : feat.shape[1], :] = feat
                return padded

            padded_ssl_feats = self._time_call("pad", _pad_all_layers)
        else:
            padded_ssl_feats = self._time_call(
                "pad", lambda: pad_sequence(ssl_feats, batch_first=True)
            )

        if self.feature_type in ["fdmpa", "all_layers_fdmpa"]:
            hc_feats = self._time_call(
                "handcrafted_extract",
                lambda: [
                    self._extract_handcrafted_features(wav, int(sr))
                    for wav, sr in zip(wavs, sample_rates)
                ],
            )
            hc_feats = self._time_call(
                "resample",
                lambda: [
                    resample_sequence_to_length(
                        hc_feat,
                        ssl_feat.shape[1]
                        if self.feature_type == "all_layers_fdmpa"
                        else ssl_feat.shape[0],
                    )
                    for ssl_feat, hc_feat in zip(ssl_feats, hc_feats)
                ],
            )
            padded_hc = self._time_call(
                "pad", lambda: pad_sequence(hc_feats, batch_first=True)
            )
            self._report_timing_if_needed()
            return (
                list(paths),
                labels_tensor,
                padded_ssl_feats,
                padded_hc,
                None,
            )

        if cluster_idxs is not None:
            padded_cluster_idxs = self._time_call(
                "pad",
                lambda: pad_sequence(
                    cluster_idxs, batch_first=True, padding_value=-1
                ),
            )
        else:
            padded_cluster_idxs = None
        self._report_timing_if_needed()
        return list(paths), labels_tensor, padded_ssl_feats, padded_cluster_idxs


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
            on_the_fly_features=kwargs.get("on_the_fly_features", True),
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
            on_the_fly_features=kwargs.get("on_the_fly_features", True),
        )
