#!/bin/bash

export all_proxy=socks5://192.168.112.1:7890 
export CUDA_VISIBLE_DEVICES=0,1,2,3

PYTHON_SCRIPT="experiments/retrieval_attack.py"

TARGET_DATASET="nq" # nq, hotpotqa
DEVICE=2
RETRIEVER="simcse" # contriever simcse ance

TARGET_THRESHOLD=4

LOG_PATH="logs/main_result/${RETRIEVER}/${TARGET_DATASET}"
mkdir -p "$LOG_PATH"

python -u $PYTHON_SCRIPT --target_dataset $TARGET_DATASET --retriever $RETRIEVER --device $DEVICE #> $LOG_PATH/bge-small-all.log  2>&1 &