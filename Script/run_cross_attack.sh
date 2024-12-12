#!/bin/bash

export all_proxy=socks5://192.168.112.1:7890 
export CUDA_VISIBLE_DEVICES=1,2,3

PYTHON_SCRIPT="Experiments/cross_attack_main.py"
TARGET_DATASET="nq" # msmarco, hotpotqa
DEVICE=1
RETRIEVER="contriever-msmarco" # contriever contriever-msmarco
# TARGET_THRESHOLD=4,9,19,49
LOG_PATH="Logs/main_result/${RETRIEVER}/${TARGET_DATASET}"
mkdir -p "$LOG_PATH"

python -u $PYTHON_SCRIPT --target_dataset $TARGET_DATASET --retriever $RETRIEVER --device $DEVICE > $LOG_PATH/1213_c2cm.log  2>&1 &