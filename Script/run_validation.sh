#!/bin/bash

export all_proxy=socks5://192.168.112.1:7890 

PYTHON_SCRIPT="Experiments/validation_all.py"

LOG_PATH="./Result/validation_all"
mkdir -p "$LOG_PATH"

python -u $PYTHON_SCRIPT > "$LOG_PATH/validation_all_test.log" 2>&1