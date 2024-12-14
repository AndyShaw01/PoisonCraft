#!/bin/bash

export all_proxy=socks5://192.168.112.1:7890 
# export CUDA_VISIBLE_DEVICES=1,2,3

PYTHON_SCRIPT="Experiments/cross_attack_main_simcse.py"
TARGET_DATASET="nq" # nq, hotpotqa
DEVICE=2
RETRIEVER="contriever" # contriever contriever-msmarco
# TARGET_THRESHOLD=4,9,19,49
LOG_PATH="Logs/transfer_attack/${RETRIEVER}2simcse/${TARGET_DATASET}"
mkdir -p "$LOG_PATH"

python -u $PYTHON_SCRIPT --target_dataset $TARGET_DATASET --retriever $RETRIEVER --device $DEVICE > $LOG_PATH/baseline_test.log  2>&1 &