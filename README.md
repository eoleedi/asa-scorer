# Fluency Scorer

## Introduction
It's my implementation for speech fluency assessment model. 
The idea for this model is from the paper [An ASR-Free Fluency Scoring Approach with Self-Supervised Learning](<https://arxiv.org/abs/2302.09928>) (Wei Liu, Kaiqi Fu, Xiaohai Tian, Shuju Shi, Wei Li, Zejun Ma, Tan Lee) proposed in the ICASSP 2023.

These implementations are unofficial, and there might be some bugs that I missed.

But, the repo will complete as soon as possible.

## Data
The SpeechOcean762 dataset used in my work is an open dataset licenced with CC BY 4.0. 
If You have downloaded speechocean762 for yourself, you can create a `.env` file and define the `SPEECHOCEAN_DIR` environment variable.

## Directions for The Programs
### The Input Features and Labels
The input generation program are in `prep_data`.
Just run the shell script in `prep_data`.
```
cd prep_data
./run.sh
```
- The labels are fluency scores in speechocean762.
- The acoustic features are extracted by **HuBert_Large**, where the dim is the value of 1024.
- The feats and labels files are collected in `data`.
- The cluster model is trained in `train_kmeans.py`, the model will be saved in `exp/kmeans`, which is used in fluency_scoring training later. 
- `kmeans_metric.py` is used to take a look the performance of kmeans clustering.

【**Noted**】: Force alignment result to replace the Kmeans predicted results

You can run the following programming if you want to try the Force alignment results for the replacement of cluster ID. 
```
python3 gen_ctc_force_align.py
```
If you choose this for the resource of cluster ID, you need to update the `run.sh`: make the `**cluster_pred=False**`

### Train Models for Fluency Scorer
- version for no cluster_id feature:
```
./noclu_run.sh
```
- version with cluster_id feature:
```
./run.sh
```

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

