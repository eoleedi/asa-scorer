#!/bin/bash
# Preparation pipeline for HuggingFace datasets (e.g., ezai-championship2023)

# Configuration
DATASET_NAME=${1:-"eoleedi/ezai-championship2023"}
TRAIN_SPLIT=${2:-"test"}  # For ezai-champ2023, use "test" for both
TEST_SPLIT=${3:-"test"}
OUTPUT_DIR="./hf_exports/$(basename ${DATASET_NAME})"
FEAT_DIR="../data"

echo "=========================================="
echo "Preparing HuggingFace dataset: ${DATASET_NAME}"
echo "Train split: ${TRAIN_SPLIT}"
echo "Test split: ${TEST_SPLIT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="

# Step 1: Export HuggingFace dataset to SO762-compatible format
echo ""
echo "Step 1: Exporting dataset to SO762 format..."
python3 prep_hf_dataset.py ${DATASET_NAME} \
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
python3 gen_seq_acoustic_feat.py ${OUTPUT_DIR} --feat_dir ${FEAT_DIR}

if [ $? -ne 0 ]; then
    echo "Error: Failed to extract features"
    exit 1
fi

# Step 3: Train k-means clustering model
echo ""
echo "Step 3: Training k-means clustering..."
python3 train_kmeans.py ${OUTPUT_DIR} --feat_dir ${FEAT_DIR}

if [ $? -ne 0 ]; then
    echo "Error: Failed to train k-means"
    exit 1
fi

# Step 4: Evaluate k-means clustering
echo ""
echo "Step 4: Evaluating k-means clustering..."
python3 kmeans_metric.py ${OUTPUT_DIR} --feat_dir ${FEAT_DIR}

echo ""
echo "=========================================="
echo "Data preparation complete!"
echo "Features saved to: ${FEAT_DIR}"
echo "K-means model saved to: ../exp/kmeans/"
echo "=========================================="
