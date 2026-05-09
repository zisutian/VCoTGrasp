accelerate launch \
    --config_file ../accelerate_configs/train_4gpu.yaml \
    ../main.py \
    --train-config grasp_anything_mlp.json
