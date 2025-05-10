#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

PYTHON_SCRIPT="experiments/target_attack.py"

RETRIEVER="simcse" # contriever simcse
DATASET="nq" # msmarco, hotpotqa nq
TOPK=4
DEVICE=3

# MODEL_CONFIG_PATH="gpt-4o-mini-2024-07-18" # deepseek-reasoner 
MODEL_CONFIG_PATH="deepseek-reasoner"
LOG_DISPLAY_K=$((TOPK + 1))

LOG_PATH="./logs/main_result/attack/${RETRIEVER}/${DATASET}/${MODEL_CONFIG_PATH}-top$LOG_DISPLAY_K/"


mkdir -p "$LOG_PATH"

# python -u $PYTHON_SCRIPT --retriever $RETRIEVER --eval_dataset $DATASET --top_k $TOPK --baseline_method $BASELINE --ablation_method $ABLATION  --url $URL --model_config_path $MODEL_CONFIG_PATH --device $DEVICE --defense_method $DEFENSE > "$LOG_PATH/top5_pd.log" 2>&1 &
python -u $PYTHON_SCRIPT --retriever $RETRIEVER --eval_dataset $DATASET --top_k $TOPK  --model_config_path $MODEL_CONFIG_PATH --device $DEVICE  #> "$LOG_PATH/top5.log" 2>&1 &