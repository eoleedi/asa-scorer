#!/bin/bash
# Preparation pipeline for HuggingFace datasets (e.g., ezai-championship2023)

set -e

# Get the directory of this script and the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Assuming structure: root/src/prep_data/run.sh
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Change to project root
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

PREP_DIR="src/prep_data"

# Configuration
DATASET_NAME=${1:-"eoleedi/ezai-championship2023"}
# ...existing code...
TRAIN_SPLIT=${2:-"test"}  # For ezai-champ2023, use "test" for both
TEST_SPLIT=${3:-"test"}
OUTPUT_DIR="data/$(basename ${DATASET_NAME})/export"
FEAT_DIR="data/$(basename ${DATASET_NAME})"

echo "=========================================="
# ...existing code...
echo "Preparing HuggingFace dataset: ${DATASET_NAME}"
echo "Train split: ${TRAIN_SPLIT}"
echo "Test split: ${TEST_SPLIT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Running from project root: ${PROJECT_ROOT}"
echo "=========================================="

# Step 1: Export HuggingFace dataset to SO762-compatible format
echo ""
echo "Step 1: Exporting dataset to SO762 format..."
python3 ${PREP_DIR}/prep_hf_dataset.py ${DATASET_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --train_split ${TRAIN_SPLIT} \
    --test_split ${TEST_SPLIT} \
    --aspects fluency prosodic

if [ $? -ne 0 ]; then
    echo "Error: Failed to export dataset"
    exit 1
fi

# Step 2: Extract acoustic features
echo ""
echo "Step 2: Extracting acoustic features..."
python3 ${PREP_DIR}/gen_seq_acoustic_feat.py ${OUTPUT_DIR} --feat_dir ${FEAT_DIR}

if [ $? -ne 0 ]; then
    echo "Error: Failed to extract features"
    exit 1
fi

# Step 3: Train k-means clustering model
echo ""
echo "Step 3: Training k-means clustering..."
python3 ${PREP_DIR}/train_kmeans.py ${OUTPUT_DIR} --feat_dir ${FEAT_DIR}

if [ $? -ne 0 ]; then
    echo "Error: Failed to train k-means"
    exit 1
fi

# Step 4: Evaluate k-means clustering
echo ""
echo "Step 4: Evaluating k-means clustering..."
python3 ${PREP_DIR}/kmeans_metric.py ${OUTPUT_DIR} --feat_dir ${FEAT_DIR}

# Step 5: Generate Proxy Targets
echo ""
echo "Step 5: Generating Proxy Targets..."
python3 ${PREP_DIR}/gen_proxy_targets.py \
    --dataset_name ${DATASET_NAME} \
    --split ${TRAIN_SPLIT} \
    --output_dir "data/proxy_targets"

echo ""
echo "=========================================="
echo "Data preparation complete!"
echo "Features saved to: ${FEAT_DIR}"
echo "K-means model saved to: exp/kmeans/"
echo "Proxy Targets saved to: data/proxy_targets/"
echo "=========================================="
