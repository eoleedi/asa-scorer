"""Stage 2: Extract HuBERT features from audio files."""

import torch
import torchaudio
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Union, Dict
from tqdm import tqdm
import numpy as np

from ..utils.data_utils import load_wav_scp, AudioDataset, save_pickle, resolve_audio_path


def extract_features(
    dataset_dir: Union[str, Path],
    feat_dir: Union[str, Path],
    split: str = "train",
    device: str = "cuda",
    batch_size: int = 1,
    layer: int = 14,
) -> Dict[str, torch.Tensor]:
    """
    Extract HuBERT-Large features from audio files.
    
    Args:
        dataset_dir: Dataset directory containing train/test subdirectories
        feat_dir: Output directory for features
        split: 'train' or 'test'
        device: Device to use ('cuda' or 'cpu')
        batch_size: Batch size for processing (default: 1)
        layer: Which HuBERT layer to extract (default: 14)
    
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
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Loading HuBERT-Large model on {device}...")
    model = torchaudio.pipelines.HUBERT_LARGE.get_model().to(device)
    
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
    extract_feat_tensor = torch.cat(extract_feat_list, dim=1)
    extract_feat_tensor = extract_feat_tensor.view(extract_feat_tensor.size(1), -1)
    print(f"Total features shape: {extract_feat_tensor.shape}")
    
    return saved_tensor_dict


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract HuBERT features")
    parser.add_argument("dataset_dir", type=str, help="Dataset directory")
    parser.add_argument("--feat_dir", type=str, default="../data", help="Feature output directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Split to process")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--layer", type=int, default=14, help="HuBERT layer to extract")
    
    args = parser.parse_args()
    
    extract_features(
        dataset_dir=args.dataset_dir,
        feat_dir=args.feat_dir,
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
        layer=args.layer,
    )
