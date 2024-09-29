#!/bin/bash

export all_proxy=socks5://192.168.112.1:7890 

PYTHON_SCRIPT="Experiments/validation_all.py"

python -u $PYTHON_SCRIPT