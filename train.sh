accelerate launch --config_file accelerate_configs/train_4gpu.yaml main.py \
    --save-root checkpoints/sr \
    --tensorboard-root checkpoints/tb \
    --run-name run1 \
    --use-bbox \
    --action-head MLP \
    --train-image-projector \
    --train-embeddings \
    --train-lm \
    --train-dataset grasp_anything \
    --train-epoch 3 \
    --train-batch-size-per-gpu 8 \
    --eval-batch-size-per-gpu 8 \
    --lr 2e-5 \
    --weight-decay 0 \
    --eval-every-n-steps 2000 \
    --save-every-n-steps 2000 \
    --bbox-ratio 0.5 \
#--load-checkpoint-dir \