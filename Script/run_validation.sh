#!/bin/bash

export all_proxy=socks5://192.168.112.1:7890 

PYTHON_SCRIPT="Experiments/validation_debug.py"

LOG_PATH="./Result/validation_debug"

mkdir -p "$LOG_PATH"

python -u $PYTHON_SCRIPT > "$LOG_PATH/validation_pr_0-5.log" 2>&1