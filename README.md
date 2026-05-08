使用lmdb重写了train.py和eval_cli.py

checkpoint内应该有pretrained文件夹存放paligemma2预训练模型,可以有vcot文件夹存放训练好的vcot模型用于load_checkpoints
checkpoints文件夹存放训练时的模型检查点


---
问题:
Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes,
but the current dtype in Gemma2ForCausalLM is torch.float32.
已解决

TODO:
解决载入模型和eval的冲突,争取直接使用checkpoint进行eval
考虑模型保存问题