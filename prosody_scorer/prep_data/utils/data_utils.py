"""Common data utilities for preparation pipeline."""

import pickle
import json
import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Union, Dict, Any


def load_wav_scp(scp_path: Union[str, Path]) -> np.ndarray:
    """
    Load wav.scp file and extract audio file paths.
    
    Args:
        scp_path: Path to wav.scp file
    
    Returns:
        Array of audio file paths
    """
    data = np.loadtxt(scp_path, delimiter=",", dtype=str)
    # Extract path from "utt_id\tpath" format
    paths = np.array([line.split("\t")[1] for line in data])
    return paths


def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """Save data to pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load data from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(data: Dict, filepath: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(filepath: Union[str, Path]) -> Dict:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def resolve_audio_path(path: str, dataset_dir: Path, split: str) -> str:
    """
    Resolve audio file path considering different formats.
    
    Args:
        path: Path from wav.scp (can be absolute, relative, or relative to split dir)
        dataset_dir: Base dataset directory
        split: Split name ('train' or 'test')
    
    Returns:
        Resolved absolute path to audio file
    """
    if os.path.isabs(path):
        return path
    elif os.path.exists(path):
        return path
    else:
        # Try relative to split directory
        return str(dataset_dir / split / path)


class AudioDataset(Dataset):
    """
    Generic audio dataset for loading wav paths and labels.
    
    Args:
        wav_paths: Array of audio file paths
        labels: Array of labels (numpy array or torch tensor)
    """
    
    def __init__(self, wav_paths: np.ndarray, labels: np.ndarray):
        self.wav_paths = wav_paths
        if isinstance(labels, np.ndarray):
            self.labels = torch.tensor(labels, dtype=torch.float)
        else:
            self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.wav_paths[idx], self.labels[idx]
