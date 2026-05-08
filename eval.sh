
python eval_cli.py \
    --load-checkpoint-dir checkpoint/vcot \
    --test-split all \
    --use-bbox \
    --action-head MLP \
    --visualize-dir results/visualize/ \
    --result-dir results/ \
    --device cuda:0 \
