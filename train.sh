accelerate launch \
    --config_file accelerate_configs/train_4gpu.yaml \
    main.py \
    --train-config train_configs/grasp_anything_mlp.json
