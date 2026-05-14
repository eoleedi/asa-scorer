# CrossAttnHCSSLScorer Training Script

This script trains a `CrossAttnHCSSLScorer` model that performs cross-attention fusion between SSL (Self-Supervised Learning) features and handcrafted (HC) acoustic features for speech scoring tasks (fluency and prosodic assessment).

## Overview

The `CrossAttnHCSSLScorer` model aligns SSL features (e.g., from emotion2vec, HuBERT) with handcrafted features (pitch, energy, etc.) using cross-attention mechanisms without MINE (Mutual Information Neural Estimation).

## Script Location

```bash
scripts/run_crossattn_hcssl.sh
```

## Usage

### Basic Run (Default: Fluency on SpeechOcean762)

```bash
cd /share/homes/eoleedi/nas169/prosody_scorer
./scripts/run_crossattn_hcssl.sh
```

### Customizing Parameters

Edit the script to modify these hyperparameters:

```bash
# Learning rate
lr=1e-3

# Batch size
batch_size=25

# Hidden dimension (embedding dimension)
hidden_dim=32

# Number of epochs
num_epochs=50

# GPU index (for CUDA_VISIBLE_DEVICES)
gpu_index=0

# Depth of cross-attention layers (number of attention blocks)
depth=2

# Number of attention heads
num_heads=4

# Dropout probability
dropout_prob=0.1

# Aspect to train: 'fluency', 'prosodic', or space-separated for multi-task
aspect="fluency"
```

### Training Different Aspects

**Fluency Only:**
```bash
aspect="fluency"
```

**Prosodic Only:**
```bash
aspect="prosodic"
```

**Multi-task (Fluency + Prosodic):**
```bash
aspect="fluency prosodic"
```

### Changing GPU

```bash
gpu_index=1  # Use GPU 1
```

Or set directly:
```bash
export CUDA_VISIBLE_DEVICES=1
```

## Dataset Requirements

The script expects the following structure in `data/speechocean762/`:
- `tr_feats_768d.pkl` - Training SSL features (768-dim)
- `te_feats_768d.pkl` - Test SSL features (768-dim)
- `tr_handcrafted_feats.pkl` - Training handcrafted features (50-dim)
- `te_handcrafted_feats.pkl` - Test handcrafted features (50-dim)
- `tr_label_utt.npy` - Training labels
- `te_label_utt.npy` - Test labels

## Model Architecture

```
CrossAttnHCSSLScorer
├── SSL Projection (768 → hidden_dim)
├── HC Projection (50 → hidden_dim)
├── Cross-Attention Fusion (depth layers)
│   ├── SSL attends to HC
│   └── HC attends to SSL
├── Attentive Stats Pooling (mean + std)
└── Scoring Head (2*hidden_dim → num_aspects)
```

### Key Architecture Parameters

- **`ssl_input_dim`**: 768 (from emotion2vec/HuBERT features)
- **`hc_input_dim`**: 50 (handcrafted features)
- **`hidden_dim`**: Common embedding dimension after projection
- **`num_heads`**: Number of attention heads in cross-attention
- **`depth`**: Number of cross-attention fusion layers
- **`dropout_prob`**: Dropout in attention and FFN layers

## Output

Results are saved in:
```
exp/SpeechOcean762/CrossAttnHCSSLScorer_<aspect>_crossattn/<lr>-<depth>-<batch_size>-<hidden_dim>-CrossAttnHCSSLScorer-br/
├── 0/
│   ├── checkpoint_best.pth        # Best model
│   ├── checkpoint_last.pth        # Final model
│   ├── train.log                  # Training log
│   └── result.csv                 # Training/test metrics per epoch
└── summary.json                   # Summary across all repeats
```

## Typical Results (SpeechOcean762)

| Task | Best Test PCC | Final Test PCC | Epochs |
|------|---------------|----------------|--------|
| Fluency | 0.793 | 0.754 | 50 |
| Prosodic | 0.785 | 0.745 | 50 |

## Advanced Usage

### Run Multiple Seeds

Modify the script to run multiple random seeds:

```bash
repeat_list=(0 1 2)  # Run 3 times with different seeds
```

### Dry Run (Check Configuration)

Add `echo` before `python3` command to preview the training call:
```bash
echo python3 -m prosody_scorer.train \
    --lr ${lr} \
    ...
```

### Resume Training

Copy a checkpoint and modify the training command in the script to load it:
```bash
--checkpoint <path_to_checkpoint>
```

## Model Comparison

**CrossAttnHCSSLScorer** vs Other Models:

- **ClusterScorer**: Uses cluster embeddings; simpler but less flexible
- **FDMPAScorer**: Uses MINE for MI estimation; more complex, branch-based
- **TransformerScorer**: Self-attention only; doesn't fuse HC features
- **CrossAttnHCSSLScorer**: Cross-attention fusion of SSL+HC; balanced complexity & performance

## Debugging

If training fails:

1. **Check data**: Verify SSL and handcrafted feature files exist
2. **GPU memory**: Reduce `batch_size` or `hidden_dim`
3. **Device**: Ensure CUDA is available or set `use_device='cpu'`
4. **Dependencies**: Check that `prosody_scorer` module is installed

## Related Scripts

- `run.sh` - ClusterScorer baseline
- `run_ezai-champ2023.sh` - EzAI Championship 2023 dataset
- `run_so762_emotion2vec.sh` - SpeechOcean762 with emotion2vec features
- `test_so762.sh` - Test on SpeechOcean762

## References

- Paper: [An ASR-Free Fluency Scoring Approach with Self-Supervised Learning](https://arxiv.org/abs/2302.09928)
- Model code: `prosody_scorer/models/scorer.py` - `CrossAttnHCSSLScorer` class (line 401)
