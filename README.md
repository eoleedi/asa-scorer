# ASA Scorer

## Introduction
It's my implementation for speech fluency assessment model. 
The idea for this model is from the paper [An ASR-Free Fluency Scoring Approach with Self-Supervised Learning](<https://arxiv.org/abs/2302.09928>) (Wei Liu, Kaiqi Fu, Xiaohai Tian, Shuju Shi, Wei Li, Zejun Ma, Tan Lee) proposed in the ICASSP 2023.

These implementations are unofficial, and there might be some bugs that I missed.

But, the repo will complete as soon as possible.

## Data
### SpeechOcean762
The SpeechOcean762 dataset used in my work is an open dataset licenced with CC BY 4.0. 
If You have downloaded speechocean762 for yourself, you can create a `.env` file and define the `SPEECHOCEAN_DIR` environment variable.

### HuggingFace Datasets
The project now supports training and testing on HuggingFace datasets, including:
- **ezai-championship2023** (`eoleedi/ezai-championship2023`)
- Any other compatible dataset with audio and fluency/prosodic scores

No manual download required - datasets are automatically fetched from HuggingFace.

## Setup
1. Create Conda environment
    ```
    conda create -n asa_scorer python=3.11
    ```
2. Install requirements 
    ```
    conda activate asa_scorer
    pip install -r requirements.txt
    ```

### Building `torbi` for Unsupported PyTorch Versions

The PyPI release of `torbi` only ships prebuilt binaries up to PyTorch 2.8.
If you are using PyTorch 2.9 (or any other version not covered), you must build
the native extension yourself and drop it into the installed package.

**Prerequisites:** CUDA toolkit matching your `torch+cuXXX` build (e.g. CUDA 12.8 for `cu128`), and `uv`.

```bash
# 1. Clone torbi source
git clone https://github.com/maxrmorrison/torbi.git ~/torbi
cd ~/torbi

# 2. Create an isolated build environment with the exact torch version you need
uv venv --python 3.11
uv pip install torch==2.9.0 torchaudio==2.9.0 \
    --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install build "setuptools<70" numpy ninja

# 3. Build the native extension (.so)
#    Adjust TORCH_CUDA_ARCH_LIST to match your GPU compute capability
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.7;8.9;9.0"
FORCE_CUDA=1 .venv/bin/python -m build --wheel --no-isolation
rm -rf build
USE_CUDA=1 .venv/bin/python build_ext_standalone.py
#    The compiled library will be at: build/_C.pt29cu128.abi3.so

# 4. Copy the .so into the project's torbi installation
TORBI_PKG=$(python -c "import torbi, os; print(os.path.dirname(torbi.__file__))")
cp build/_C.pt29cu128.abi3.so "$TORBI_PKG/"
```

Verify the fix:
```bash
python -c "import torbi; print('torbi OK')"
```

> **Note:** Repeat step 3–4 whenever you upgrade PyTorch to a version that has
> no prebuilt binary on PyPI. Substitute the version strings (e.g. `pt29cu128`)
> accordingly.

## Inference
1. Download the pretrained model (audio and kmeans model)
    https://drive.google.com/drive/folders/1439B6_JRJbmr_zB2PWBSYHRwJDbxGDrP?usp=sharing
2. Activate the conda environment
    ```
    conda activate asa_scorer
    ```
3. Run Inference
    ```
    python inference.py path/to/audio \
        --kmeans_model path/to/kmeans_model.joblib \
        --checkpoint path/to/asa-scorer-cluster_fluency+prosodic.pth \ 
        --aspect fluency prosodic
    ```

It will predict the score accordingly.

Specifically, it produce something like the below with the flency and prosodic score in order between 0-2.

    Prediction(0-2): [[0.8962262 0.9108325]]

## Directions for The Programs
### The Input Features and Labels

#### For SpeechOcean762 Dataset

Directly download and process from HuggingFace:
```bash
./scripts/prep_so762_hf.sh
```

#### For ezai-championship2023 Dataset
Use the dedicated preparation script:
```bash
./scripts/prep_ezai_champ2023.sh
```

#### For Other HuggingFace Datasets
```bash
./scripts/prep_hf_dataset.sh "dataset-name/dataset-id" "train_split" "test_split"
```

**What these scripts do:**
- Download dataset from HuggingFace (if using HF datasets)
- Export to local format compatible with the training pipeline
- Extract **HuBERT_Large** acoustic features (dim=1024)
- Train K-means clustering model
- Evaluate clustering quality
- All outputs saved to `data/` and `exp/kmeans/`

See [prosody_scorer/prep_data/README.md](prosody_scorer/prep_data/README.md) for detailed documentation.

【**Noted**】: Force alignment result to replace the Kmeans predicted results

You can run the following programming if you want to try the Force alignment results for the replacement of cluster ID. 
```
python3 gen_ctc_force_align.py
```
If you choose this for the resource of cluster ID, you need to update the `run.sh`: make the `**cluster_pred=False**`

### Train Models for Fluency Scorer

#### For SpeechOcean762
- version for no cluster_id feature:
```bash
./noclu_run.sh
```
- version with cluster_id feature:
```bash
./run.sh
```

#### For ezai-championship2023
```bash
./run_ezai-champ2023.sh
```

This will train on the full dataset (using "test" split for both training and validation).

#### For Other Datasets
Modify `run_ezai-champ2023.sh` and change the dataset parameters.

## Results
### SpeechOcean762
| Model             | Fluency PCC | Prosodic PCC |
|-------------------|:-----------:|:------------:|
| ClusterScorer     |    0.79     |     0.80     |

### Ezai-championship2023 (OOD)
| Model             | Fluency PCC | Prosodic PCC |
|-------------------|:-----------:|:------------:|
| ClusterScorer     |   0.352    |    0.346    |
| + Window Sliding  |   0.370    |    0.463    |

