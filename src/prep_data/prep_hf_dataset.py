#!/usr/bin/env python3
"""
Export HuggingFace datasets to SO762-compatible format for prep pipeline.

This script downloads a HuggingFace dataset and exports it to:
- wav.scp files (maps utterance_id to audio file path)
- Audio WAV files saved locally
- Label files (same format as SO762)
"""

import os
import argparse
import json
import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


def export_hf_dataset(dataset_name, output_dir, train_split="train", test_split="test", aspects=None):
    """
    Export HuggingFace dataset to SO762-compatible format.
    
    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "eoleedi/ezai-championship2023")
        output_dir: Directory to save exported files
        train_split: Name of training split
        test_split: Name of test split
        aspects: List of aspect names to extract (default: ["accuracy", "completeness", "fluency", "prosodic", "total"])
    """
    if aspects is None:
        aspects = ["accuracy", "completeness", "fluency", "prosodic", "total"]
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create wav directories
    train_wav_dir = os.path.join(train_dir, "wav")
    test_wav_dir = os.path.join(test_dir, "wav")
    os.makedirs(train_wav_dir, exist_ok=True)
    os.makedirs(test_wav_dir, exist_ok=True)
    
    print(f"Loading HuggingFace dataset: {dataset_name}")
    print(f"Train split: {train_split}, Test split: {test_split}")
    
    # Load dataset splits
    train_data = load_dataset(dataset_name, split=train_split)
    test_data = load_dataset(dataset_name, split=test_split)
    
    # Export function
    def export_split(data, split_name, split_dir, wav_dir):
        wav_scp_path = os.path.join(split_dir, "wav.scp")
        labels = []
        utt2score = {}
        
        with open(wav_scp_path, "w") as wav_scp:
            for idx, item in enumerate(tqdm(data, desc=f"Exporting {split_name}")):
                # Create utterance ID
                utt_id = f"{split_name}_{idx:06d}"
                
                # Save audio file
                audio = item["audio"]
                array = audio["array"]
                sr = int(audio["sampling_rate"])
                wav_path = os.path.join(wav_dir, f"{utt_id}.wav")
                sf.write(wav_path, array, sr)
                
                # Write to wav.scp (relative path from split_dir)
                relative_wav_path = f"wav/{utt_id}.wav"
                wav_scp.write(f"{utt_id}\t{relative_wav_path}\n")
                
                # Extract labels
                label_row = []
                score_dict = {}
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
        
        # Save labels as numpy array
        labels_array = np.array(labels)
        return labels_array, utt2score
    
    # Export both splits
    train_labels, train_utt2score = export_split(train_data, "train", train_dir, train_wav_dir)
    test_labels, test_utt2score = export_split(test_data, "test", test_dir, test_wav_dir)
    
    # Save labels to data directory (parent of output_dir)
    data_dir = os.path.dirname(output_dir)
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "tr_label_utt.npy"), train_labels)
    np.save(os.path.join(data_dir, "te_label_utt.npy"), test_labels)
    
    # Save scores.json (combined for compatibility)
    all_utt2score = {**train_utt2score, **test_utt2score}
    scores_json_path = os.path.join(output_dir, "scores.json")
    with open(scores_json_path, "w") as f:
        json.dump(all_utt2score, f, indent=2)
    
    print(f"\nExport complete!")
    print(f"  Train samples: {len(train_labels)}")
    print(f"  Test samples: {len(test_labels)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Labels saved to: {data_dir}/tr_label_utt.npy, {data_dir}/te_label_utt.npy")
    print(f"  Scores saved to: {scores_json_path}")
    
    return output_dir


def main():
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
        args.output_dir,
        args.train_split,
        args.test_split,
        args.aspects
    )


if __name__ == "__main__":
    main()
