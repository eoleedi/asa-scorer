#!/bin/bash
set -x
source .env
stage=1
stop_stage=1000

lr=1e-3
batch_size=25
hidden_dim=32
num_epochs=50
gpu_index=0
# use_device='cpu'
use_device='cuda'
depth=3
num_heads=1
dataset="eoleedi/ezai-championship2023"
train_split="train"  # Use test split for training (full dataset)
test_split="train"   # Use test split for testing (same as training)

model=ClusterScorer
model(){
  NonClusterScorer
  ClusterScorer
  TransformerScorer
}

aspect="fluency prosodic"
tag_aspect=${aspect// /+}
tag=ezai-champ2023_${tag_aspect}Score_trainontest
# fluency prosodic

exp_dir=exp/${tag}/${lr}-${depth}-${batch_size}-${hidden_dim}-${model}-br

# repeat times
repeat_list=(0)
seed_list=(0 11 22 33 44)

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    for repeat in "${repeat_list[@]}"; do
        mkdir -p $exp_dir/${repeat}
        python3 train.py \
            --lr ${lr} \
            --exp-dir ${exp_dir}/${repeat} \
            --batch_size ${batch_size} --hidden_dim ${hidden_dim} \
            --model ${model} --n-epochs ${num_epochs} --use_device ${use_device} --gpu_index ${gpu_index} \
            --depth ${depth} --num_heads ${num_heads} --dataset ${dataset} \
            --train_split ${train_split} --test_split ${test_split} \
			--seed "${seed_list[$repeat]}" --aspect ${aspect}
    done
    python3 collect_summary.py --exp-dir $exp_dir
    exit 0
fi
