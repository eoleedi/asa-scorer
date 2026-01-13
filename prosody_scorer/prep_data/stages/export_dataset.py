"""Stage 1: Export HuggingFace dataset to local format."""

import os
import numpy as np
import soundfile as sf
from datasets import load_dataset
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

from ..utils.data_utils import save_json


def export_hf_dataset(
    dataset_name: str,
    output_dir: Path,
    train_split: str = "train",
    test_split: str = "test",
    aspects: Optional[List[str]] = None,
) -> Path:
    """
    Export HuggingFace dataset to SO762-compatible format.
    
    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "eoleedi/ezai-championship2023", "mispeech/speechocean762")
        output_dir: Directory to save exported files
        train_split: Name of training split
        test_split: Name of test split
        aspects: List of aspect names to extract (default: ["accuracy", "completeness", "fluency", "prosodic", "total"])
    
    Returns:
        Path to output directory
    """
    if aspects is None:
        aspects = ["accuracy", "completeness", "fluency", "prosodic", "total"]
    
    output_dir = Path(output_dir)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Create wav directories
    train_wav_dir = train_dir / "wav"
    test_wav_dir = test_dir / "wav"
    train_wav_dir.mkdir(exist_ok=True)
    test_wav_dir.mkdir(exist_ok=True)
    
    print(f"Loading HuggingFace dataset: {dataset_name}")
    print(f"Train split: {train_split}, Test split: {test_split}")
    
    # Load dataset splits
    train_data = load_dataset(dataset_name, split=train_split)
    test_data = load_dataset(dataset_name, split=test_split)
    
    # Detect dataset type and adjust processing
    is_speechocean762 = "speechocean762" in dataset_name.lower()
    
    # Export function
    def export_split(data, split_name: str, split_dir: Path, wav_dir: Path) -> Tuple[np.ndarray, dict]:
        """Export a single split of the dataset."""
        wav_scp_path = split_dir / "wav.scp"
        labels = []
        utt2score = {}
        
        with open(wav_scp_path, "w") as wav_scp:
            for idx, item in enumerate(tqdm(data, desc=f"Exporting {split_name}")):
                # Create utterance ID
                if is_speechocean762 and "id" in item:
                    # Use original ID from SpeechOcean762
                    utt_id = str(item["id"])
                else:
                    utt_id = f"{split_name}_{idx:06d}"
                
                # Save audio file
                audio = item["audio"]
                array = audio["array"]
                sr = int(audio["sampling_rate"])
                wav_path = wav_dir / f"{utt_id}.wav"
                sf.write(wav_path, array, sr)
                
                # Write to wav.scp (relative path from split_dir)
                relative_wav_path = f"wav/{utt_id}.wav"
                wav_scp.write(f"{utt_id}\t{relative_wav_path}\n")
                
                # Extract labels
                label_row = []
                score_dict = {}
                
                # Handle SpeechOcean762 nested structure
                if is_speechocean762 and "scores" in item:
                    scores = item["scores"]
                    for aspect in aspects:
                        if aspect in scores:
                            value = float(scores[aspect])
                        else:
                            value = 0.0
                        label_row.append(value)
                        score_dict[aspect] = value
                else:
                    # Handle flat structure (ezai-championship2023, etc.)
                    for aspect in aspects:
                        if aspect in item:
                            value = float(item[aspect])
                            label_row.append(value)
                            score_dict[aspect] = value
                        else:
                            # If aspect doesn't exist in dataset, use 0
                            label_row.append(0.0)
                            score_dict[aspect] = 0.0
                
                labels.append(label_row)
                utt2score[utt_id] = score_dict
        
        labels_array = np.array(labels)
        return labels_array, utt2score
    
    # Export both splits
    train_labels, train_utt2score = export_split(train_data, "train", train_dir, train_wav_dir)
    test_labels, test_utt2score = export_split(test_data, "test", test_dir, test_wav_dir)
    
    # Save labels to data directory (parent of output_dir)
    data_dir = output_dir.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "tr_label_utt.npy", train_labels)
    np.save(data_dir / "te_label_utt.npy", test_labels)
    
    # Save scores.json (combined for compatibility)
    all_utt2score = {**train_utt2score, **test_utt2score}
    scores_json_path = output_dir / "scores.json"
    save_json(all_utt2score, scores_json_path)
    
    print(f"\nExport complete!")
    print(f"  Train samples: {len(train_labels)}")
    print(f"  Test samples: {len(test_labels)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Labels saved to: {data_dir}/tr_label_utt.npy, {data_dir}/te_label_utt.npy")
    print(f"  Scores saved to: {scores_json_path}")
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export HuggingFace dataset to SO762-compatible format"
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="HuggingFace dataset name (e.g., eoleedi/ezai-championship2023)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./hf_dataset_export",
        help="Output directory for exported files"
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Name of training split"
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
        help="Name of test split"
    )
    parser.add_argument(
        "--aspects",
        nargs="+",
        default=["accuracy", "completeness", "fluency", "prosodic", "total"],
        help="Aspects to extract from dataset"
    )
    
    args = parser.parse_args()
    
    export_hf_dataset(
        args.dataset,
        Path(args.output_dir),
        args.train_split,
        args.test_split,
        args.aspects,
    )
