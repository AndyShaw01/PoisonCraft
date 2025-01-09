#!/bin/bash

BASH_SCRIPT="./scripts/run_poisoncraft.sh"
DOMAIN_LIST="1,2,3,4,5,6,7,8,9,10,11,12,13,14" # ,2,3,4,5,6,7,8,9,10,11,12,13,14"
ADV_LENGTH_LIST="55,65,75,85" #,55,60,65,70,75,80,85"
GPU_LIST="0,1,2,3"
for DOMAIN_ID in $(echo $DOMAIN_LIST | sed "s/,/ /g")
do
    for ADV_LENGTH in $(echo $ADV_LENGTH_LIST | sed "s/,/ /g")
    do
        if [ "$ADV_LENGTH" = "50" ]; then
            GPU_NUM=0
        elif [ "$ADV_LENGTH" = "55" ]; then
            GPU_NUM=0
        elif [ "$ADV_LENGTH" = "60" ]; then
            GPU_NUM=1
        elif [ "$ADV_LENGTH" = "65" ]; then
            GPU_NUM=1
        elif [ "$ADV_LENGTH" = "70" ]; then
            GPU_NUM=2
        elif [ "$ADV_LENGTH" = "75" ]; then
            GPU_NUM=2
        elif [ "$ADV_LENGTH" = "80" ]; then
            GPU_NUM=3
        elif [ "$ADV_LENGTH" = "85" ]; then
            GPU_NUM=3
        fi
        bash $BASH_SCRIPT $DOMAIN_ID $ADV_LENGTH $GPU_NUM &
        echo "Domain $DOMAIN_ID is running on GPU $GPU_NUM"
    done
done

# Wait for all background tasks to complete
wait
