#!/bin/bash
# Wrapper script for new pipeline - maintains backward compatibility
# Usage: ./run_prep_ezai-champ2023_new.sh [--stage N] [--stop-stage M]

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

# Dataset configuration
DATASET_NAME="eoleedi/ezai-championship2023"
TRAIN_SPLIT="train"
TEST_SPLIT="train"
OUTPUT_DIR="data/ezai-championship2023/ezai-champ2023"
FEAT_DIR="data/ezai-championship2023"
MODEL_DIR="exp/kmeans/ezai-championship2023"

echo "Running NEW modular pipeline from project root: $PROJECT_ROOT"

# Use the new pipeline.py
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
echo "✓ NEW pipeline completed!"
echo ""
echo "Next step: Run training with run_ezai-champ2023.sh"
echo "=========================================="
