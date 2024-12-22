#!/bin/bash

PYTHON_SCRIPT="Experiments/data_process.py"

LOG_PATH="Logs/tmp_data/"

TOPK=99

mkdir -p "$LOG_PATH"

python -u "$PYTHON_SCRIPT" --k $TOPK > "$LOG_PATH/get_ground_truth.log" 2>&1