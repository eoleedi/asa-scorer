#!/bin/bash
set -x

stage=1
stop_stage=1000

lr=1e-3
batch_size=25
hidden_dim=32
num_epochs=50
gpu_index=0  # GPU to use (via CUDA_VISIBLE_DEVICES)
# use_device='cpu'
use_device='cuda'
depth=3
num_heads=1
# SO762_dir=${SPEECHOCEAN762_DIR}
load_cluster_index=True
dataset_type='so762'

# Set GPU device
export CUDA_VISIBLE_DEVICES=${gpu_index}
model=ClusterScorer
model(){
  NonClusterScorer
  ClusterScorer
  TransformerScorer
}


aspect="prosodic"
tag_aspect=${aspect// /+}
tag=SSLfeat_${tag_aspect}Score
# acc cpn flu psd ttl

exp_dir=exp/${tag}/${lr}-${depth}-${batch_size}-${hidden_dim}-${model}-br

# repeat times
repeat_list=(0)
seed_list=(0 11 22 33 44)

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    for repeat in "${repeat_list[@]}"; do
        mkdir -p $exp_dir/${repeat}
        python3 -m prosody_scorer.train \
            --lr ${lr} \
            --exp-dir ${exp_dir}/${repeat} \
            --batch_size ${batch_size} --hidden_dim ${hidden_dim} \
            --model ${model} --n-epochs ${num_epochs} --use_device ${use_device} \
            --depth ${depth} --num_heads ${num_heads} --dataset_type ${dataset_type} --load_cluster_index ${load_cluster_index} \
			--seed "${seed_list[$repeat]}" --aspect ${aspect} \
            ${extra_args}
    done
    python3 -m prosody_scorer.collect_summary --exp-dir $exp_dir
    exit 0
fi
