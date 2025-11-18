#!/bin/bash
# Preparation pipeline specifically for ezai-championship2023 dataset
# This uses the "test" split for both training and testing (full dataset)
# Usage:
#   ./run_prep_ezai-champ2023.sh [--stage N] [--stop-stage M]
# Examples:
#   ./run_prep_ezai-champ2023.sh            # run all stages (1..4)
#   ./run_prep_ezai-champ2023.sh --stage 2  # start from stage 2 and run through 4
#   ./run_prep_ezai-champ2023.sh --stage 2 --stop-stage 3  # run only stages 2 and 3

set -euo pipefail

# source .env

DATASET_NAME="eoleedi/ezai-championship2023"
TRAIN_SPLIT="train"
TEST_SPLIT="train"
OUTPUT_DIR="../data/ezai-championship2023/ezai-champ2023"
FEAT_DIR="../data/ezai-championship2023"

# default stages
STAGE=1
STOP_STAGE=4

usage() {
    echo "Usage: $0 [--stage N] [--stop-stage M]"
    echo "  --stage N       Start from stage N (default: ${STAGE})"
    echo "  --stop-stage M  Stop at stage M (default: ${STOP_STAGE})"
}

# parse args (simple long-option parser)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --stop-stage)
            STOP_STAGE="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# validate numeric
re='^[0-9]+$'
if ! [[ $STAGE =~ $re && $STOP_STAGE =~ $re ]]; then
    echo "Error: --stage and --stop-stage must be integers"
    exit 1
fi
if [ "$STAGE" -lt 1 ] || [ "$STOP_STAGE" -lt "$STAGE" ]; then
    echo "Error: invalid stage range: ${STAGE}..${STOP_STAGE}"
    exit 1
fi

echo "=========================================="
echo "Preparing ezai-championship2023 dataset"
echo "Stages to run: ${STAGE} .. ${STOP_STAGE}"
echo "=========================================="

# helper to check whether to run a stage
should_run() {
    local s="$1"
    if [ "$STAGE" -le "$s" ] && [ "$STOP_STAGE" -ge "$s" ]; then
        return 0
    else
        return 1
    fi
}

# Stage 1: Export HuggingFace dataset to SO762-compatible format
if should_run 1; then
    echo ""
    echo "Step 1: Exporting dataset..."
    python3 prep_hf_dataset.py ${DATASET_NAME} \
        --output_dir ${OUTPUT_DIR} \
        --train_split ${TRAIN_SPLIT} \
        --test_split ${TEST_SPLIT} \
        --aspects fluency prosodic
else
    echo "Skipping Step 1 (export dataset)"
fi

# Stage 2: Extract acoustic features
if should_run 2; then
    echo ""
    echo "Step 2: Extracting HuBERT features..."
    python3 gen_seq_acoustic_feat.py ${OUTPUT_DIR} --feat_dir ${FEAT_DIR}
else
    echo "Skipping Step 2 (extract features)"
fi

# Stage 3: Train k-means clustering model
if should_run 3; then
    echo ""
    echo "Step 3: Training k-means (50 clusters)..."
    python3 train_kmeans.py ${OUTPUT_DIR} --feat_dir ${FEAT_DIR}
else
    echo "Skipping Step 3 (train k-means)"
fi

# Stage 4: Evaluate k-means clustering
if should_run 4; then
    echo ""
    echo "Step 4: Evaluating clustering quality..."
    python3 kmeans_metric.py ${OUTPUT_DIR} --feat_dir ${FEAT_DIR}
else
    echo "Skipping Step 4 (evaluate clustering)"
fi

echo ""
echo "=========================================="
echo "✓ Data preparation complete!"
echo ""
echo "Generated files (if corresponding stages were run):"
echo "  - Labels: ${FEAT_DIR}/tr_label_utt.npy, ${FEAT_DIR}/te_label_utt.npy"
echo "  - Features: ${FEAT_DIR}/tr_feats.pkl, ${FEAT_DIR}/te_feats.pkl"
echo "  - Clusters: ${FEAT_DIR}/tr_cluster_index.pkl, ${FEAT_DIR}/te_cluster_index.pkl"
echo "  - K-means model: ../exp/kmeans/kmeans_model.joblib"
echo ""
echo "Next step: Run training with run_ezai-champ2023.sh"
echo "=========================================="
