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
load_cluster_index=True

# Set GPU device
export CUDA_VISIBLE_DEVICES=${gpu_index}

model=NonClusterScorer
model(){
  NonClusterScorer
  ClusterScorer
  TransformerScorer
}

aspect="fluency prosodic"
tag_aspect=${aspect// /+}
tag=SpeechOcean762_${tag_aspect}Score_non_cluster
# acc cpn flu psd ttl

exp_dir=exp/SpeechOcean762/${tag}/${lr}-${depth}-${batch_size}-${hidden_dim}-${model}-br

# repeat times
repeat_list=(0 1 2 3 4)
seed_list=(0 11 22 33 44)

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    for repeat in "${repeat_list[@]}"; do
        mkdir -p $exp_dir/${repeat}
        python3 -m prosody_scorer.train \
            --lr ${lr} \
            --exp-dir ${exp_dir}/${repeat} \
            --batch_size ${batch_size} --hidden_dim ${hidden_dim} \
            --model ${model} --n-epochs ${num_epochs} --use_device ${use_device} \
            --depth ${depth} --num_heads ${num_heads} --load_cluster_index ${load_cluster_index} \
			--seed "${seed_list[$repeat]}" --aspect ${aspect} \
            ${extra_args}
    done
    python3 -m prosody_scorer.collect_summary --exp-dir $exp_dir
    exit 0
fi
