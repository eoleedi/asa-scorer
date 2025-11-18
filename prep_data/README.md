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
3. Extract HuBERT acoustic features
4. Train k-means clustering (50 clusters)
5. Generate cluster indices

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

### 3. Feature Extraction
**Script:** `gen_seq_acoustic_feat.py`

Extracts HuBERT-Large (layer 14) features from all audio files.

```bash
python3 gen_seq_acoustic_feat.py /path/to/dataset --feat_dir ../data
```

Outputs:
- `../data/tr_feats.pkl` - Training features dictionary
- `../data/te_feats.pkl` - Test features dictionary

### 4. K-Means Clustering
**Script:** `train_kmeans.py`

Trains MiniBatch K-Means on training features (50 clusters by default).

```bash
python3 train_kmeans.py /path/to/dataset --feat_dir ../data --output_dir ../exp/kmeans
```

Outputs:
- `../exp/kmeans/kmeans_model.joblib` - Trained k-means model
- `../data/tr_cluster_index.pkl` - Training cluster assignments
- `../data/te_cluster_index.pkl` - Test cluster assignments

### 5. Clustering Evaluation
**Script:** `kmeans_metric.py`

Evaluates k-means clustering quality.

```bash
python3 kmeans_metric.py /path/to/dataset --feat_dir ../data
```

## Output Files

After running the preparation pipeline, you'll have:

```
data/
├── tr_label_utt.npy          # Training labels (N x 5): [acc, cpn, flu, psd, ttl]
├── te_label_utt.npy          # Test labels
├── tr_feats.pkl              # Training HuBERT features (dict: path -> features)
├── te_feats.pkl              # Test HuBERT features
├── tr_cluster_index.pkl      # Training cluster assignments
├── te_cluster_index.pkl      # Test cluster assignments
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
- **Feature extraction** requires CUDA for speed (falls back to CPU if unavailable)
- **K-means training** may take several minutes on large datasets
- All scripts use HuBERT-Large layer 14 features (1024 dimensions)

## Troubleshooting

**Out of memory during feature extraction:**
- Reduce batch size or process in smaller chunks
- Use CPU instead of GPU (slower but uses less memory)

**Missing dependencies:**
```bash
pip install soundfile scipy datasets
```

**Dataset not found:**
- Check HuggingFace dataset name and split names
- Ensure you have internet connection for downloading datasets
