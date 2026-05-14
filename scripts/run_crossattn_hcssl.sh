#!/bin/bash
set -x

stage=1
stop_stage=1000

# Training hyperparameters
lr=1e-3
batch_size=25
hidden_dim=32
num_epochs=50
gpu_index=0  # GPU to use (via CUDA_VISIBLE_DEVICES)
use_device='cuda'
depth=2
num_heads=4
dropout_prob=0.1

# Dataset and model configuration
data_dir="data/speechocean762"
dataset_type='so762'
model='CrossAttnHCSSLScorer'

# Set GPU device
export CUDA_VISIBLE_DEVICES=${gpu_index}

# Aspect to train on (can be 'fluency', 'prosodic', or space-separated list for multi-task)
aspect="fluency"
tag_aspect=${aspect// /+}
tag=CrossAttnHCSSLScorer_${tag_aspect}_crossattn

exp_dir=exp/SpeechOcean762/${tag}/${lr}-${depth}-${batch_size}-${hidden_dim}-${model}-br

# Repeat times and seeds
repeat_list=(0)
seed_list=(0 11 22 33 44)

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "========================================="
    echo "Training CrossAttnHCSSLScorer"
    echo "Aspect: ${aspect}"
    echo "Data: ${data_dir}"
    echo "LR: ${lr}, Depth: ${depth}, Batch: ${batch_size}, Hidden: ${hidden_dim}"
    echo "Num Heads: ${num_heads}, Dropout: ${dropout_prob}"
    echo "========================================="
    
    for repeat in "${repeat_list[@]}"; do
        mkdir -p $exp_dir/${repeat}
        python3 -m prosody_scorer.train \
            --lr ${lr} \
            --data_dir ${data_dir} \
            --exp-dir ${exp_dir}/${repeat} \
            --batch_size ${batch_size} \
            --hidden_dim ${hidden_dim} \
            --model ${model} \
            --n-epochs ${num_epochs} \
            --use_device ${use_device} \
            --depth ${depth} \
            --num_heads ${num_heads} \
            --dataset_type ${dataset_type} \
            --dropout_prob ${dropout_prob} \
            --seed "${seed_list[$repeat]}" \
            --aspect ${aspect} \
            ${extra_args}
    done
    
    # Collect summary statistics
    python3 -m prosody_scorer.collect_summary --exp-dir $exp_dir
    exit 0
fi
