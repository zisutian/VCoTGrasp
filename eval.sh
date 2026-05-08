
python eval_cli.py \
    --load-checkpoint-dir ../VCoT-Grasp-self/checkpoint/vcot \
    --test-split all \
    --use-bbox \
    --action-head MLP \
    --visualize-dir results/visualize/ \
    --result-dir results/ \
    --device cuda:0 \
