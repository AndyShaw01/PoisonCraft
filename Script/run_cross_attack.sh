#!/bin/bash

PYTHON_SCRIPT="Experiments/cross_attack.py"

LOG_PATH="Logs/cross_attack"
mkdir -p "$LOG_PATH"

export CUDA_VISIBLE_DEVICES=1

python -u $PYTHON_SCRIPT > $LOG_PATH/recheck.log 2>&1 &