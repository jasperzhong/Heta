#!/bin/bash

MODEL=${1:-"rgcn"}
DATASET=${2:-"donor"}
PART_METHOD=${3:-"metis"}
PREDICT_CATEGORY=${4:-"Project"}
N_CLASSES=${5:-"2"}
BATCH_SIZE=${6:-"64"}
CACHE_METHOD=${7:-"none"}
EMBEDDING_SIZE=${8:-"64"}
FANOUT=${9:-"25,20"}

cmd="rm -rf /dev/shm/*; /opt/conda/envs/jasper/bin/python3 dgl/train_dist.py --graph_name ${DATASET} --model ${MODEL} --ip_config dgl/ip_config.txt --num_epochs 1 --batch_size ${BATCH_SIZE} --n_classes ${N_CLASSES} --predict_category ${PREDICT_CATEGORY} --eval_every 1 --fan_out ${FANOUT}  --num_hidden ${EMBEDDING_SIZE} --num_gpus 8 --dgl-sparse --cache-method ${CACHE_METHOD}"

python3 ~/repos/dgl/tools/launch.py \
    --workspace /home/ubuntu/repos/Heta \
    --num_trainers 8 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config partitions/dgl/${DATASET}_${PART_METHOD}/${DATASET}.json \
    --ip_config dgl/ip_config.txt \
    "${cmd}" > dgl_${MODEL}_${DATASET}_${PART_METHOD}_${CACHE_METHOD}_${EMBEDDING_SIZE}.log 2>&1 