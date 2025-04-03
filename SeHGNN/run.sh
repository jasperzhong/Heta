#!/bin/bash

N_GPU=${1:-8}

# N_GPU == 1
if [ $N_GPU -eq 1 ]; then
    python main.py \
    --stages 200 \
    --num-hops 2 \
    --label-feats \
    --num-label-hops 2 \
    --n-layers-1 2 \
    --n-layers-2 2 \
    --hidden 256 \
    --residual \
    --act leaky_relu \
    --bns \
    --label-bns \
    --lr 0.001 \
    --weight-decay 0 \
    --threshold 0.75 \
    --patience 200 \
    --gama 10 \
    --amp \
    --seeds 1 \
    > single_GPU_ogbn-mag_sehgnn_256.log 2>&1 
else
OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=${N_GPU} main_dist.py \
    --stages 200 \
    --num-hops 2 \
    --label-feats \
    --num-label-hops 2 \
    --n-layers-1 2 \
    --n-layers-2 2 \
    --hidden 256 \
    --residual \
    --act leaky_relu \
    --bns \
    --label-bns \
    --lr 0.001 \
    --weight-decay 0 \
    --threshold 0.75 \
    --patience 200 \
    --gama 10 \
    --amp \
    --seeds 1 \
    > ${N_GPU}GPUs_ogbn-mag_sehgnn_256_dev4.log 2>&1 
fi

