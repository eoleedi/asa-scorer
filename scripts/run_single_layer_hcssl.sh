#!/bin/bash
set -euo pipefail
set -x

stage=1
stop_stage=1000

lr=1e-3
batch_size=16
hidden_dim=64
num_epochs=50
gpu_index=0
use_device='cuda'
dataset_type='so762'
data_dir='data/speechocean762'

model='SingleLayerHCSSLScorer'
aspect='prosodic'
hc_aux_weight=0.1
fdmpa_num_tokens=16
dropout_prob=0.1

export CUDA_VISIBLE_DEVICES=${gpu_index}

tag_aspect=${aspect// /+}
tag=SingleLayerHCSSLScorer_${tag_aspect}_hc_aux${hc_aux_weight}
exp_dir=exp/SpeechOcean762/${tag}/${lr}-${batch_size}-${hidden_dim}

repeat_list=(0)
seed_list=(0 11 22 33 44)

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "========================================="
    echo "Training SingleLayerHCSSLScorer"
    echo "Aspect: ${aspect}"
    echo "Data: ${data_dir}"
    echo "LR: ${lr}, Batch: ${batch_size}, Hidden: ${hidden_dim}"
    echo "HC aux weight: ${hc_aux_weight}, Tokens: ${fdmpa_num_tokens}"
    echo "========================================="

    for repeat in "${repeat_list[@]}"; do
        mkdir -p "$exp_dir/${repeat}"
        python3 -m prosody_scorer.train \
            --lr ${lr} \
            --exp-dir "$exp_dir/${repeat}" \
            --batch_size ${batch_size} \
            --hidden_dim ${hidden_dim} \
            --model ${model} \
            --n-epochs ${num_epochs} \
            --use_device ${use_device} \
            --dataset_type ${dataset_type} \
            --data_dir ${data_dir} \
            --seed "${seed_list[$repeat]}" \
            --aspect ${aspect} \
            --fdmpa_num_tokens ${fdmpa_num_tokens} \
            --hc_aux_weight ${hc_aux_weight} \
            --dropout_prob ${dropout_prob} \
            ${extra_args:-}
    done

    python3 -m prosody_scorer.collect_summary --exp-dir "$exp_dir"
fi
