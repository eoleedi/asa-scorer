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

