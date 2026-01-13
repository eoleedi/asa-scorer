#!/usr/bin/env python3
"""
Data preparation pipeline for fluency scoring datasets.

This pipeline orchestrates the complete data preparation workflow:
1. Export HuggingFace dataset to local format (optional)
2. Extract HuBERT features from audio
3. Train K-Means clustering model
4. Evaluate clustering quality

Usage:
    python pipeline.py <dataset_dir> [options]
    
Examples:
    # Run all stages for ezai-championship2023
    python pipeline.py data/ezai-championship2023/ezai-champ2023 \\
        --feat_dir data/ezai-championship2023 \\
        --output_dir exp/kmeans/ezai-championship2023
    
    # Run only stages 2-3 (feature extraction and kmeans)
    python pipeline.py data/ezai-championship2023/ezai-champ2023 \\
        --stage 2 --stop_stage 3
    
    # Export HuggingFace dataset and run full pipeline
    python pipeline.py data/ezai-championship2023/ezai-champ2023 \\
        --stage 0 \\
        --hf_dataset eoleedi/ezai-championship2023 \\
        --train_split train --test_split train
"""

import argparse
from pathlib import Path
import sys

from .stages import (
    export_hf_dataset,
    extract_features,
    train_kmeans_model,
    evaluate_clustering,
)


def main():
    parser = argparse.ArgumentParser(
        description="Data preparation pipeline for fluency scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Required arguments
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Dataset directory (should contain train/ and test/ subdirectories)",
    )
    
    # Directory arguments
    parser.add_argument(
        "--feat_dir",
        type=str,
        default="../data",
        help="Directory for features and labels (default: ../data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../exp/kmeans",
        help="Directory for trained models (default: ../exp/kmeans)",
    )
    
    # Stage control
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        help="Starting stage (0=export HF dataset, 1=extract features, 2=train kmeans, 3=evaluate) (default: 1)",
    )
    parser.add_argument(
        "--stop_stage",
        type=int,
        default=3,
        help="Stopping stage (default: 3)",
    )
    
    # Stage 0: HuggingFace dataset export (optional)
    parser.add_argument(
        "--hf_dataset",
        type=str,
        help="HuggingFace dataset name (e.g., eoleedi/ezai-championship2023). Required if stage=0.",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Train split name for HF dataset (default: train)",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
        help="Test split name for HF dataset (default: test)",
    )
    parser.add_argument(
        "--aspects",
        nargs="+",
        default=["accuracy", "completeness", "fluency", "prosodic", "total"],
        help="Score aspects to extract (default: accuracy completeness fluency prosodic total)",
    )
    
    # Stage 1: Feature extraction
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for feature extraction (default: cuda)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=14,
        help="HuBERT layer to extract (default: 14)",
    )
    
    # Stage 2: K-Means training
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=50,
        help="Number of clusters for K-Means (default: 50)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum K-Means iterations (default: 100)",
    )
    parser.add_argument(
        "--kmeans_batch_size",
        type=int,
        default=10000,
        help="MiniBatch K-Means batch size (default: 10000)",
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=20,
        help="Number of K-Means initializations (default: 20)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    
    args = parser.parse_args()
    
    # Convert paths
    dataset_dir = Path(args.dataset_dir)
    feat_dir = Path(args.feat_dir)
    output_dir = Path(args.output_dir)
    
    # Validate stage range
    if args.stage < 0 or args.stop_stage < args.stage or args.stop_stage > 3:
        print("Error: Invalid stage range. Stage must be 0-3 and stop_stage >= stage.")
        sys.exit(1)
    
    # Validate HF dataset requirement (allow None for backward compatibility with local datasets)
    if args.stage <= 0 <= args.stop_stage and not args.hf_dataset:
        print("Warning: --hf_dataset not specified. Assuming dataset is already in local format.")
        print("         To export from HuggingFace, provide --hf_dataset (e.g., mispeech/speechocean762)")
    
    # Auto-detect dataset type if hf_dataset is provided
    if args.hf_dataset:
        dataset_name_lower = args.hf_dataset.lower()
        if "speechocean762" in dataset_name_lower:
            print(f"Detected SpeechOcean762 dataset: {args.hf_dataset}")
        elif "ezai" in dataset_name_lower or "championship" in dataset_name_lower:
            print(f"Detected EZAI Championship dataset: {args.hf_dataset}")
        else:
            print(f"Processing custom dataset: {args.hf_dataset}")
    
    print("=" * 70)
    print("Data Preparation Pipeline")
    print("=" * 70)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Feature directory: {feat_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Stages to run:     {args.stage} -> {args.stop_stage}")
    print("=" * 70)
    
    # Stage 0: Export HuggingFace dataset
    if args.stage <= 0 <= args.stop_stage:
        if args.hf_dataset:
            print("\n" + "=" * 70)
            print("STAGE 0: Exporting HuggingFace dataset")
            print("=" * 70)
            export_hf_dataset(
                dataset_name=args.hf_dataset,
                output_dir=dataset_dir,
                train_split=args.train_split,
                test_split=args.test_split,
                aspects=args.aspects,
            )
        else:
            print("\n" + "=" * 70)
            print("STAGE 0: Skipped (no HuggingFace dataset specified)")
            print("=" * 70)
            print("Assuming dataset is already in local format at:", dataset_dir)
    
    # Stage 1: Extract features
    if args.stage <= 1 <= args.stop_stage:
        print("\n" + "=" * 70)
        print("STAGE 1: Extracting HuBERT features")
        print("=" * 70)
        
        print("\nExtracting TRAINING features...")
        extract_features(
            dataset_dir=dataset_dir,
            feat_dir=feat_dir,
            split="train",
            device=args.device,
            layer=args.layer,
        )
        
        print("\nExtracting TEST features...")
        extract_features(
            dataset_dir=dataset_dir,
            feat_dir=feat_dir,
            split="test",
            device=args.device,
            layer=args.layer,
        )
    
    # Stage 2: Train K-Means
    if args.stage <= 2 <= args.stop_stage:
        print("\n" + "=" * 70)
        print("STAGE 2: Training K-Means clustering")
        print("=" * 70)
        train_kmeans_model(
            dataset_dir=dataset_dir,
            feat_dir=feat_dir,
            output_dir=output_dir,
            n_clusters=args.n_clusters,
            max_iter=args.max_iter,
            batch_size=args.kmeans_batch_size,
            n_init=args.n_init,
            random_state=args.random_state,
        )
    
    # Stage 3: Evaluate clustering
    if args.stage <= 3 <= args.stop_stage:
        print("\n" + "=" * 70)
        print("STAGE 3: Evaluating clustering quality")
        print("=" * 70)
        evaluate_clustering(
            dataset_dir=dataset_dir,
            feat_dir=feat_dir,
            model_dir=output_dir,
        )
    
    print("\n" + "=" * 70)
    print("✓ Pipeline completed successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  - Labels:           {feat_dir}/tr_label_utt.npy, {feat_dir}/te_label_utt.npy")
    print(f"  - Features:         {feat_dir}/tr_feats.pkl, {feat_dir}/te_feats.pkl")
    print(f"  - Cluster indices:  {feat_dir}/tr_cluster_index.pkl, {feat_dir}/te_cluster_index.pkl")
    print(f"  - K-means model:    {output_dir}/kmeans_model.joblib")
    print("\nNext step: Run training with your training script")
    print("=" * 70)


if __name__ == "__main__":
    main()
