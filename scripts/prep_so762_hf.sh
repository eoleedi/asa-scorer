#!/bin/bash
# Preparation pipeline for SpeechOcean762 dataset from HuggingFace
# This script downloads and processes the dataset directly from mispeech/speechocean762
# Usage:
#   ./prep_so762_hf.sh

set -euo pipefail

# Get the directory of this script and the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Configuration
HF_DATASET="mispeech/speechocean762"
OUTPUT_DIR="data/speechocean762/so762"
FEAT_DIR="data/speechocean762"
MODEL_DIR="exp/kmeans/so762"
TRAIN_SPLIT="train"
TEST_SPLIT="test"

echo "=========================================="
echo "SpeechOcean762 HuggingFace Pipeline"
echo "=========================================="
echo "HuggingFace dataset: ${HF_DATASET}"
echo "Output directory:    ${OUTPUT_DIR}"
echo "Feature directory:   ${FEAT_DIR}"
echo "Model directory:     ${MODEL_DIR}"
echo "=========================================="

# Run full pipeline from stage 0 (export) to stage 3 (evaluate)
CUDA_VISIBLE_DEVICES=0 python3 -m prosody_scorer.prep_data.pipeline "${OUTPUT_DIR}" \
    --feat_dir "${FEAT_DIR}" \
    --output_dir "${MODEL_DIR}" \
    --stage 1 \
    --stop_stage 3 \
    --hf_dataset "${HF_DATASET}" \
    --train_split "${TRAIN_SPLIT}" \
    --test_split "${TEST_SPLIT}" \
    "$@"

echo ""
echo "=========================================="
echo "✓ SpeechOcean762 HuggingFace preparation complete!"
echo ""
echo "Generated files:"
echo "  - Dataset:        ${OUTPUT_DIR}/"
echo "  - Labels:         ${FEAT_DIR}/tr_label_utt.npy, ${FEAT_DIR}/te_label_utt.npy"
echo "  - Features:       ${FEAT_DIR}/tr_feats.pkl, ${FEAT_DIR}/te_feats.pkl"
echo "  - Cluster index:  ${FEAT_DIR}/tr_cluster_index.pkl, ${FEAT_DIR}/te_cluster_index.pkl"
echo "  - K-means model:  ${MODEL_DIR}/kmeans_model.joblib"
echo ""
echo "Next step: Run training with your training script"
echo "=========================================="
