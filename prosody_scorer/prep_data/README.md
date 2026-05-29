# Data Preparation Scripts

This directory contains scripts for preparing datasets for fluency scoring model training.

## Supported Datasets

1. **SpeechOcean762** - Local dataset with wav.scp format
2. **HuggingFace Datasets** - Including ezai-championship2023 and others

## Quick Start

### For ezai-championship2023 Dataset

```bash
cd prep_data
bash run_prep_ezai-champ2023.sh
```

This will:
1. Download the dataset from HuggingFace
2. Export to wav files and labels
3. Train k-means clustering from HuBERT features extracted on the fly

### For SpeechOcean762 Dataset

```bash
cd prep_data
export SPEECHOCEAN762_DIR=/path/to/speechocean762
bash run.sh
```

### For Other HuggingFace Datasets

```bash
cd prep_data
bash run_prep_hf.sh "dataset-name/dataset-id" "train_split" "test_split"
```

Example:
```bash
bash run_prep_hf.sh "eoleedi/ezai-championship2023" "test" "test"
```

## Pipeline Steps

### 1. Data Export (HuggingFace only)
**Script:** `prep_hf_dataset.py`

Exports HuggingFace datasets to SO762-compatible format:
- Audio files saved as WAV
- `wav.scp` mapping utterance IDs to file paths
- Label files (.npy format)
- `scores.json` with all utterance scores

```bash
python3 prep_hf_dataset.py eoleedi/ezai-championship2023 \
    --output_dir ./hf_exports/ezai-champ2023 \
    --train_split test \
    --test_split test \
    --aspects fluency prosodic
```

### 2. Label Generation (SO762 only)
**Script:** `gen_seq_data_utt.py`

For SO762, reads wav.scp and scores.json to create label arrays.

```bash
python3 gen_seq_data_utt.py /path/to/speechocean762 scores.json
```

### 3. K-Means Clustering
**Script:** `train_kmeans.py`

Trains MiniBatch K-Means from HuBERT-Large features extracted on the fly from
audio batches. It does not write feature or cluster-index pickles.

```bash
python3 train_kmeans.py /path/to/dataset \
    --feat_dir ../data \
    --output_dir ../exp/kmeans \
    --feature_batch_size 8
```

Outputs:
- `../exp/kmeans/kmeans_model.joblib` - Trained k-means model
- `../data/cluster_centers.pkl` - K-means cluster centers

### 4. Clustering Evaluation
**Script:** `evaluate.py`

Evaluates k-means clustering quality by extracting HuBERT features on the fly.

```bash
python3 evaluate.py /path/to/dataset --feat_dir ../data --model_dir ../exp/kmeans
```

## Output Files

After running the preparation pipeline, you'll have:

```
data/
├── tr_label_utt.npy          # Training labels (N x 5): [acc, cpn, flu, psd, ttl]
├── te_label_utt.npy          # Test labels
└── cluster_centers.pkl       # K-means cluster centers

exp/kmeans/
└── kmeans_model.joblib       # Trained k-means model

hf_exports/                   # (HuggingFace datasets only)
└── ezai-champ2023/
    ├── train/
    │   ├── wav.scp
    │   └── wav/              # Audio files
    ├── test/
    │   ├── wav.scp
    │   └── wav/
    └── scores.json
```

## Notes

- **ezai-championship2023**: Uses "test" split for both training and testing (full dataset)
- **On-the-fly feature extraction** requires CUDA for speed (falls back to CPU if unavailable)
- **K-means training** may take several minutes on large datasets
- All scripts use HuBERT-Large layer 14 features (1024 dimensions)

## Troubleshooting

**Out of memory during feature extraction:**
- Reduce `--feature_batch_size`
- Use CPU instead of GPU (slower but uses less memory)

**Missing dependencies:**
```bash
pip install soundfile scipy datasets
```

**Dataset not found:**
- Check HuggingFace dataset name and split names
- Ensure you have internet connection for downloading datasets
