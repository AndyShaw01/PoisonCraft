#!/bin/bash

export all_proxy=socks5://192.168.112.1:7890 
export CUDA_VISIBLE_DEVICES=0
PYTHON_SCRIPT="Experiments/validation_debug.py"

RETRIEVER="contriever"
DATASET="nq"
TOPK=4
LOG_PATH="./Logs/main_result/attack/${RETRIEVER}/${DATASET}/top${TOK+1}"

mkdir -p "$LOG_PATH"

python -u $PYTHON_SCRIPT --retriever $RETRIEVER --eval_dataset $DATASET --top_k $TOPK> "$LOG_PATH/main_result.log" 2>&1