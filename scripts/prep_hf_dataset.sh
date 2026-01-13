#!/bin/bash
# Preparation pipeline for HuggingFace datasets using the new modular pipeline
# Usage:
#   ./run_prep_hf.sh <dataset_name> [train_split] [test_split]
# Examples:
#   ./run_prep_hf.sh eoleedi/ezai-championship2023
#   ./run_prep_hf.sh eoleedi/ezai-championship2023 train test
#   ./run_prep_hf.sh username/my-dataset train test

set -euo pipefail

# Get the directory of this script and the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Source .env if it exists
if [ -f ".env" ]; then
    source ".env"
fi

# Configuration
DATASET_NAME=${1:-"eoleedi/ezai-championship2023"}
TRAIN_SPLIT=${2:-"train"}
TEST_SPLIT=${3:-"test"}
OUTPUT_DIR="data/$(basename ${DATASET_NAME})/export"
FEAT_DIR="data/$(basename ${DATASET_NAME})"
MODEL_DIR="exp/kmeans/$(basename ${DATASET_NAME})"

echo "=========================================="
echo "HuggingFace Dataset Preparation Pipeline"
echo "=========================================="
echo "Dataset:          ${DATASET_NAME}"
echo "Train split:      ${TRAIN_SPLIT}"
echo "Test split:       ${TEST_SPLIT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Feature directory: ${FEAT_DIR}"
echo "Model directory:   ${MODEL_DIR}"
echo "Running from:      ${PROJECT_ROOT}"
echo "=========================================="

# Run the modular pipeline
python3 -m prosody_scorer.prep_data.pipeline "${OUTPUT_DIR}" \
    --feat_dir "${FEAT_DIR}" \
    --output_dir "${MODEL_DIR}" \
    --stage 0 \
    --hf_dataset "${DATASET_NAME}" \
    --train_split "${TRAIN_SPLIT}" \
    --test_split "${TEST_SPLIT}" \
    --aspects fluency prosodic \
    "$@"

echo ""
echo "=========================================="
echo "✓ HuggingFace dataset preparation complete!"
echo ""
echo "Generated files:"
echo "  - Dataset:        ${OUTPUT_DIR}"
echo "  - Labels:         ${FEAT_DIR}/tr_label_utt.npy, ${FEAT_DIR}/te_label_utt.npy"
echo "  - Features:       ${FEAT_DIR}/tr_feats.pkl, ${FEAT_DIR}/te_feats.pkl"
echo "  - Cluster index:  ${FEAT_DIR}/tr_cluster_index.pkl, ${FEAT_DIR}/te_cluster_index.pkl"
echo "  - K-means model:  ${MODEL_DIR}/kmeans_model.joblib"
echo ""
echo "Next step: Run training with your training script"
echo "=========================================="
