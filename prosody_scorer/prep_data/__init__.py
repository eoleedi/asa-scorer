"""Data preparation modules for fluency scoring."""

from .stages import (
    export_hf_dataset,
    extract_features,
    train_kmeans_model,
    evaluate_clustering,
)

from .utils import (
    load_wav_scp,
    save_pickle,
    load_pickle,
    AudioDataset,
)

__version__ = "0.1.0"

__all__ = [
    # Stage functions
    "export_hf_dataset",
    "extract_features",
    "train_kmeans_model",
    "evaluate_clustering",
    # Utilities
    "load_wav_scp",
    "save_pickle",
    "load_pickle",
    "AudioDataset",
]
