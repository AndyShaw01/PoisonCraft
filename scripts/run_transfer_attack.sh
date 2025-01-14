#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

PYTHON_SCRIPT="TransferAttack/evaluate.py"

LOG_PATH="Logs/transfer_attack"
mkdir -p "$LOG_PATH"

python -u $PYTHON_SCRIPT  > $LOG_PATH/transferattack_evaluate_nq_allseeds.log  2>&1 &