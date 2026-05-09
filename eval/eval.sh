#!/usr/bin/env bash


CHECKPOINT_DIR=../VCoT-Grasp-self/checkpoint
TASK=vcot
CHECKPOINT= 
#epoch2_step25695_final


cd ..

python eval_cli.py \
    --load-checkpoint-dir ${CHECKPOINT_DIR}/${TASK}/${CHECKPOINT} \
    --test-split all \
    --visualize-dir results/${TASK}/${CHECKPOINT}/visualize/ \
    --result-dir results/${TASK}/${CHECKPOINT}/ \
    --device cuda:0
