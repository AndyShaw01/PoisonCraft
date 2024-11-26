#!/bin/bash

export all_proxy=socks5://192.168.112.1:7890 
export CUDA_VISIBLE_DEVICES=2

PYTHON_SCRIPT="Experiments/cross_attack_main.py"

LOG_PATH="Logs/cross_attack"
mkdir -p "$LOG_PATH"

python -u $PYTHON_SCRIPT # > $LOG_PATH/recheck.log # 2>&1 &