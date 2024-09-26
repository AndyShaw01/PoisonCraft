#!/bin/bash

BASH_SCRIPT="./Script/run_gcg_rag.sh"
# GROUP_LIST="1,2,3,4,5,6,7,8,9,10,11,12,13,14"
GROUP_NUM="1"
CONTROL_LENGTH_LIST="50,55,60,65,70,75"
GPU_LIST="0,1,2,3"
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
    elif [ "$CONTROL_LENGTH" = "60" ]; then
        GPU_NUM=2
    elif [ "$CONTROL_LENGTH" = "75" ]; then
        GPU_NUM=3
    fi
    bash $BASH_SCRIPT $GROUP_NUM $CONTROL_LENGTH $GPU_NUM &
    echo "Group $GROUP_NUM is running on GPU $GPU_NUM"
done

# 等待所有后台任务完成
wait
