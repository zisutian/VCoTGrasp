#!/usr/bin/env bash

cd ..

python eval_cli.py \
    --load-checkpoint-dir checkpoints/sr/run1/epoch2_step25695_final \
    --test-split all \
    --visualize-dir results/visualize/ \
    --result-dir results/ \
    --device cuda:0
