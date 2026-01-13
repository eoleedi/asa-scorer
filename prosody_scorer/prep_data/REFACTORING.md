# Data Preparation - Refactored Modules

## 🎯 What Changed?

The preparation pipeline has been **refactored into a modular structure** for better maintainability, reusability, and testability.

## 📁 New Structure

```
prep_data/
├── __init__.py              # Package exports
├── pipeline.py              # Main CLI entry point (NEW!)
├── utils/                   # Shared utilities (NEW!)
│   ├── __init__.py
│   └── data_utils.py        # Common data loading/saving functions
├── stages/                  # Individual pipeline stages (NEW!)
│   ├── __init__.py
│   ├── export_dataset.py    # Stage 0: HF dataset export
│   ├── extract_features.py  # Stage 1: HuBERT feature extraction
│   ├── train_kmeans.py      # Stage 2: K-Means training
│   └── evaluate.py          # Stage 3: Clustering evaluation
└── [old scripts]            # Original scripts (DEPRECATED)
```

## 🚀 Quick Start

### Option 1: Use the New Pipeline (Recommended)

```bash
cd prosody_scorer/prep_data

# Run full pipeline (stages 0-3)
python pipeline.py data/ezai-championship2023/ezai-champ2023 \
    --feat_dir data/ezai-championship2023 \
    --output_dir exp/kmeans/ezai-championship2023 \
    --stage 0 \
    --hf_dataset eoleedi/ezai-championship2023 \
    --train_split train --test_split train \
    --aspects fluency prosodic

# Run only feature extraction (stage 1)
python pipeline.py data/ezai-championship2023/ezai-champ2023 \
    --feat_dir data/ezai-championship2023 \
    --stage 1 --stop_stage 1

# Run with custom parameters
python pipeline.py data/ezai-championship2023/ezai-champ2023 \
    --feat_dir data/ezai-championship2023 \
    --n_clusters 100 \
    --device cuda
```

### Option 2: Use Individual Stage Modules

```python
from prep_data.stages import extract_features, train_kmeans_model
from pathlib import Path

# Extract features programmatically
extract_features(
    dataset_dir=Path("data/ezai-championship2023/ezai-champ2023"),
    feat_dir=Path("data/ezai-championship2023"),
    split="train",
    device="cuda",
)

# Train k-means
train_kmeans_model(
    dataset_dir=Path("data/ezai-championship2023/ezai-champ2023"),
    feat_dir=Path("data/ezai-championship2023"),
    output_dir=Path("exp/kmeans"),
    n_clusters=50,
)
```

### Option 3: Shell Script Wrapper (Backward Compatible)

```bash
# New shell script using the modular pipeline
./run_prep_ezai-champ2023_new.sh

# Old shell script (still works, but deprecated)
./run_prep_ezai-champ2023.sh
```

## ✨ Key Improvements

### 1. **No More Code Duplication**
- `load_wav_scp()`, `AudioDataset`, pickle operations → all in `utils/data_utils.py`
- Shared by all stages

### 2. **Each Stage is a Reusable Function**
- Can be imported and called from other scripts
- Easy to test in notebooks or unit tests
- Clear function signatures with type hints

### 3. **Better Parameter Management**
- Parameters passed explicitly to each function
- No global variables or hardcoded paths
- Easy to adjust for experiments

### 4. **Flexible Execution**
- Run full pipeline or individual stages
- Use CLI, import as library, or call from notebooks
- Stage control with `--stage` and `--stop_stage`

### 5. **Improved Readability**
- Clear separation of concerns
- Self-documenting code with docstrings
- Type hints for better IDE support

## 📚 Module Documentation

### `utils/data_utils.py`
Common utilities for data I/O:
- `load_wav_scp()` - Load wav.scp files
- `save_pickle()` / `load_pickle()` - Pickle file operations
- `save_json()` / `load_json()` - JSON file operations
- `AudioDataset` - PyTorch dataset for audio + labels
- `resolve_audio_path()` - Handle different path formats

### `stages/export_dataset.py`
Export HuggingFace datasets to local format:
- Function: `export_hf_dataset()`
- Saves wav files, wav.scp, labels, and scores.json

### `stages/extract_features.py`
Extract HuBERT features from audio:
- Function: `extract_features()`
- Configurable layer, device, batch size
- Saves features as pickle dictionary

### `stages/train_kmeans.py`
Train MiniBatch K-Means clustering:
- Function: `train_kmeans_model()`
- Configurable clusters, iterations, batch size
- Generates cluster assignments for train/test

### `stages/evaluate.py`
Evaluate clustering quality:
- Function: `evaluate_clustering()`
- Davies-Bouldin and Calinski-Harabasz scores
- Evaluates both train and test sets

### `pipeline.py`
Main CLI orchestrating all stages:
- Comprehensive argument parsing
- Stage control and validation
- Progress reporting

## 🔧 Migration Guide

### For Existing Scripts

If you have scripts calling the old preparation scripts:

**Old way:**
```bash
python prep_hf_dataset.py dataset --output_dir out
python gen_seq_acoustic_feat.py out --feat_dir data
python train_kmeans.py out --feat_dir data --output_dir exp
```

**New way (CLI):**
```bash
python pipeline.py out --feat_dir data --output_dir exp \
    --stage 0 --hf_dataset dataset
```

**New way (Python):**
```python
from prep_data.stages import export_hf_dataset, extract_features, train_kmeans_model
from pathlib import Path

# More explicit and testable
export_hf_dataset("dataset", Path("out"))
extract_features(Path("out"), Path("data"), split="train")
train_kmeans_model(Path("out"), Path("data"), Path("exp"))
```

### For Notebooks

```python
# Import stages as needed
from prep_data.stages import extract_features
from prep_data.utils import load_pickle

# Use directly
features = extract_features(
    dataset_dir="data/my_dataset",
    feat_dir="data/features",
    split="train",
    device="cuda",
)

# Load and inspect
feat_dict = load_pickle("data/features/tr_feats.pkl")
print(f"Loaded {len(feat_dict)} feature tensors")
```

## 🧪 Testing

Each stage module can be run independently:

```bash
# Test export
python stages/export_dataset.py eoleedi/ezai-championship2023 \
    --output_dir test_out

# Test feature extraction
python stages/extract_features.py test_out --feat_dir test_data --split train

# Test k-means
python stages/train_kmeans.py test_out --feat_dir test_data --output_dir test_exp

# Test evaluation
python stages/evaluate.py test_out --feat_dir test_data --model_dir test_exp
```

## ⚠️ Deprecation Notice

The following scripts are **deprecated** but still functional:
- `prep_hf_dataset.py` → Use `stages/export_dataset.py` or `pipeline.py`
- `gen_seq_acoustic_feat.py` → Use `stages/extract_features.py` or `pipeline.py`
- `train_kmeans.py` → Use `stages/train_kmeans.py` or `pipeline.py`
- `kmeans_metric.py` → Use `stages/evaluate.py` or `pipeline.py`

They will be removed in a future version. Please migrate to the new structure.

## 💡 Tips

1. **For quick experiments**: Use `pipeline.py` with stage control
2. **For notebooks**: Import individual stage functions
3. **For production**: Use the modular functions with proper error handling
4. **For debugging**: Run individual stage modules with `--help`

## 📝 Example Workflows

### Full Pipeline from HuggingFace
```bash
python pipeline.py data/new_dataset/export \
    --feat_dir data/new_dataset \
    --output_dir exp/kmeans/new_dataset \
    --stage 0 \
    --hf_dataset username/new-dataset \
    --n_clusters 100
```

### Re-run Only Feature Extraction
```bash
python pipeline.py data/dataset/export \
    --feat_dir data/dataset \
    --stage 1 --stop_stage 1 \
    --device cpu  # Use CPU instead
```

### Try Different K-Means Configurations
```bash
python pipeline.py data/dataset/export \
    --feat_dir data/dataset \
    --output_dir exp/kmeans_100 \
    --stage 2 --stop_stage 3 \
    --n_clusters 100 \
    --max_iter 200
```
