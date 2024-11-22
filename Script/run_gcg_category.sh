#!/bin/bash

BASH_SCRIPT="./Script/run_gcg_rag.sh"
DOMAIN_LIST="1,2,3,4,5,6,7,8,9,10,11,12,13,14"
CONTROL_LENGTH_LIST="50,55,60,65" #,70,75,80,85"
GPU_LIST="0,1,2,3"
for DOMAIN_ID in $(echo $DOMAIN_LIST | sed "s/,/ /g")
do
    for CONTROL_LENGTH in $(echo $CONTROL_LENGTH_LIST | sed "s/,/ /g")
    do
        if [ "$CONTROL_LENGTH" = "50" ]; then
            GPU_NUM=1
        elif [ "$CONTROL_LENGTH" = "55" ]; then
            GPU_NUM=2
        elif [ "$CONTROL_LENGTH" = "60" ]; then
            GPU_NUM=3
        elif [ "$CONTROL_LENGTH" = "65" ]; then
            GPU_NUM=1
        elif [ "$CONTROL_LENGTH" = "70" ]; then
            GPU_NUM=2
        elif [ "$CONTROL_LENGTH" = "75" ]; then
            GPU_NUM=3
        elif [ "$CONTROL_LENGTH" = "80" ]; then
            GPU_NUM=1
        elif [ "$CONTROL_LENGTH" = "85" ]; then
            GPU_NUM=2
        fi
        bash $BASH_SCRIPT $DOMAIN_ID $CONTROL_LENGTH $GPU_NUM &
        echo "Group $DOMAIN_ID is running on GPU $GPU_NUM"
    done
done

# 等待所有后台任务完成
wait
