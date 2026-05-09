accelerate launch \
    --config_file ../accelerate_configs/train_4gpu.yaml \
    ../train.py \
    --train-config grasp_anything_mlp.json
