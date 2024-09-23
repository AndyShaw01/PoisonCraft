#!/bin/bash

export all_proxy=socks5://192.168.112.1:7890 

PYTHON_SCRIPT="Experiments/case_study.py"

python -u $PYTHON_SCRIPT