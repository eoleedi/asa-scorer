"""Stage 2: Extract HuBERT features from audio files."""

import torch
import torchaudio
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Union, Dict
from tqdm import tqdm
import numpy as np

from ..utils.data_utils import (
    load_wav_scp,
    AudioDataset,
    save_pickle,
    resolve_audio_path,
)

from funasr import AutoModel as FunASRAutoModel


class Emotion2VecExtractor:
    """Wrapper for emotion2vec feature extraction using FunASR"""

    def __init__(
        self,
        model_id="emotion2vec/emotion2vec_plus_large",
        device="cuda",
        granularity="frame",
        layer=-1,
    ):
        self.model = FunASRAutoModel(
            model=model_id, hub="hf", device=device  # Use huggingface hub
        )
        self.granularity = granularity
        self.device = device
        self.layer = layer  # -1 means final layer, 0-7 for specific layers

    def extract_features(self, waveform, sample_rate=16000):
        """Extract features from waveform

        Args:
            waveform: numpy array of audio samples or torch tensor
            sample_rate: sample rate of the audio

        Returns:
            features: tensor of shape (seq_len, 1024) for frame-level or (1024,) for utterance-level
        """
        # Convert waveform to tensor if needed
        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform, dtype=torch.float32)

        # Ensure waveform is on the correct device
        waveform = waveform.to(self.device)

        # If we need a specific layer, use the underlying model's extract_features method
        if self.layer != -1:
            try:
                # Access the underlying emotion2vec model
                underlying_model = self.model.model

                # Call extract_features directly on the underlying model
                result = underlying_model.extract_features(
                    source=waveform,
                    padding_mask=None,
                    mask=False,
                    remove_extra_tokens=True,
                )

                # Get layer_results
                if (
                    "layer_results" in result
                    and len(result["layer_results"]) > self.layer
                ):
                    layer_feat = result["layer_results"][self.layer]
                    if isinstance(layer_feat, torch.Tensor):
                        return layer_feat.cpu(), None
                    elif isinstance(layer_feat, np.ndarray):
                        return torch.tensor(layer_feat, dtype=torch.float32), None
                else:
                    print(
                        f"Warning: Layer {self.layer} not available (only {len(result.get('layer_results', []))} layers), using final output"
                    )
                    # Fall back to final output
                    if "x" in result:
                        return result["x"].cpu(), None
            except Exception as e:
                print(
                    f"Warning: Could not extract layer {self.layer}: {e}. Using FunASR default output."
                )

        # Use FunASR's generate method for final layer (default behavior)
        result = self.model.generate(
            waveform, granularity=self.granularity, extract_embedding=True
        )

        # Get the features from result
        if isinstance(result, list) and len(result) > 0:
            feats = result[0].get("feats", None)
            if feats is not None:
                if isinstance(feats, np.ndarray):
                    return torch.tensor(feats, dtype=torch.float32), None
                return feats, None

        raise ValueError("Failed to extract features from emotion2vec")


def extract_features(
    dataset_dir: Union[str, Path],
    feat_dir: Union[str, Path],
    split: str = "train",
    device: str = "cuda",
    batch_size: int = 1,
    layer: int = 14,
    model_name: str = "hubert_large",
) -> Dict[str, torch.Tensor]:
    """
    Extract HuBERT-Large or Emotion2Vec features from audio files.

    Args:
        dataset_dir: Dataset directory containing train/test subdirectories
        feat_dir: Output directory for features
        split: 'train' or 'test'
        device: Device to use ('cuda' or 'cpu')
        batch_size: Batch size for processing (default: 1)
        layer: Which layer to extract (HuBERT: 0-23, default 14; Emotion2Vec: 0-7, default -1 for final layer)

    Returns:
        Dictionary mapping audio paths to feature tensors
    """
    dataset_dir = Path(dataset_dir)
    feat_dir = Path(feat_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Determine file prefix and split name
    prefix = "tr" if split == "train" else "te"

    # Load data
    wav_scp_path = dataset_dir / split / "wav.scp"
    label_file = feat_dir / f"{prefix}_label_utt.npy"

    print(f"Loading wav.scp from: {wav_scp_path}")
    print(f"Loading labels from: {label_file}")

    wav_paths = load_wav_scp(wav_scp_path)
    labels = np.load(label_file)

    dataset = AudioDataset(wav_paths, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    if model_name == "hubert_large":
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading HuBERT-Large model on {device}...")
        model = torchaudio.pipelines.HUBERT_LARGE.get_model().to(device)
    elif model_name == "emotion2vec":
        device_str = device if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        print(f"Loading Emotion2Vec model on {device} (layer={layer})...")
        model = Emotion2VecExtractor(device=device_str, layer=layer)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # Extract features
    print(f"Extracting features for {split} split...")
    extract_feat_list = []

    for paths, _ in tqdm(dataloader, desc=f"Extracting {split} features"):
        audio_list = []

        for path in paths:
            # Resolve audio path
            audio_path = resolve_audio_path(path, dataset_dir, split)

            # Load waveform
            waveform, sample_rate = torchaudio.load(audio_path)
            audio_list.append(waveform)

        # Pad to max length in batch
        max_length = max(waveform.size(1) for waveform in audio_list)
        padded_audio_list = [
            torch.nn.functional.pad(
                waveform.squeeze(0), (0, max_length - waveform.size(1)), mode="constant"
            ).unsqueeze(0)
            for waveform in audio_list
        ]

        # Stack and process
        audio = torch.stack(padded_audio_list, dim=0)
        audio = audio.to(device)
        audio = audio.view(audio.size(0), -1)

        # Extract features
        with torch.inference_mode():
            if model_name == "emotion2vec":
                # Emotion2vec processes one sample at a time
                batch_features = []
                for i in range(audio.size(0)):
                    single_audio = audio[i : i + 1]
                    features, _ = model.extract_features(single_audio)
                    batch_features.append(features)
                # Stack features: (batch, seq_len, hidden_dim)
                my_feature = torch.stack(batch_features, dim=0)
            else:
                # HuBERT returns a list of layers
                audio_embedding, _ = model.extract_features(audio)
                # Extract specified layer
                my_feature = audio_embedding[layer]

            extract_feat_list.append(my_feature.cpu())

    print("Creating feature dictionary...")

    # Create dictionary mapping paths to features
    saved_tensor_dict = {}
    for j, (paths, _) in enumerate(dataloader):
        for path in paths:
            if path not in saved_tensor_dict:
                saved_tensor_dict[path] = extract_feat_list[j][0]

    # Save features
    output_file = feat_dir / f"{prefix}_feats.pkl"
    save_pickle(saved_tensor_dict, output_file)

    print(f"Saved {len(saved_tensor_dict)} feature tensors to {output_file}")

    # Print shape info
    # Note: Cannot concatenate all features due to variable sequence lengths
    # Just print info about the first feature and feature dimension
    if len(extract_feat_list) > 0:
        first_feat = extract_feat_list[0]
        print(f"First feature shape: {first_feat.shape}")
        print(f"Feature dimension: {first_feat.shape[-1]}")
        print(f"Total number of utterances: {len(saved_tensor_dict)}")

    return saved_tensor_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract HuBERT features")
    parser.add_argument("dataset_dir", type=str, help="Dataset directory")
    parser.add_argument(
        "--feat_dir", type=str, default="../data", help="Feature output directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Split to process",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--layer",
        type=int,
        default=14,
        help="Layer to extract (HuBERT: 0-23, default 14; Emotion2Vec: 0-23 or -1 for final layer)",
    )
    parser.add_argument(
        "--model_name", type=str, default="hubert_large", help="Model name"
    )

    args = parser.parse_args()

    extract_features(
        dataset_dir=args.dataset_dir,
        feat_dir=args.feat_dir,
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
        layer=args.layer,
        model_name=args.model_name,
    )
