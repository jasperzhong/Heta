#!/bin/bash

MODEL=${1:-"rgcn"}
DATASET=${2:-"ogbn-mag"}
PREDICT_CATEGORY=${3:-"paper"}
N_CLASSES=${4:-"349"}
BATCH_SIZE=${5:-"512"}
NTYPES_W_FEATS=${6:-""}
CACHE_METHOD=${7:-"none"}
EMBEDDING_SIZE=${8:-"64"}

cmd="/opt/conda/envs/jasper/bin/python3  train_dist.py --graph_name ${DATASET} --model ${MODEL} --ip_config ip_config.txt --num_epochs 3 --batch_size ${BATCH_SIZE} --n_classes ${N_CLASSES} --predict_category ${PREDICT_CATEGORY} --eval_every 1 --fan_out 25,20 --num_hidden ${EMBEDDING_SIZE} --part_dir /home/ubuntu/repos/Heta/partitions/Heta/${DATASET} --num_gpus 8 --dgl-sparse --cache-method ${CACHE_METHOD} --num_layers 2"

# not freebase
if [ "$DATASET" != "freebase" ]; then
    cmd="${cmd} --ntypes-w-feats ${NTYPES_W_FEATS}"
fi
echo $cmd

# launch pytorch dist 
python3 ~/repos/dgl/tools/launch.py \
    --workspace /home/ubuntu/repos/Heta \
    --num_trainers 8 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config partitions/Heta/${DATASET}/${DATASET}.json \
    --ip_config ip_config.txt \
    "${cmd}" > Heta_${MODEL}_${DATASET}_${CACHE_METHOD}_${EMBEDDING_SIZE}.log 2>&1 