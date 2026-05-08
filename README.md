checkpoint内应该有pretrained文件夹存放paligemma2预训练模型,可以有vcot文件夹存放训练好的vcot模型用于load_checkpoints,这两部分需下载

checkpoints文件夹存放训练时的模型检查点

data存放数据集,origin_split是原作者处理后的csv文件

data_scripts为数据集下载与处理脚本,使用lmdb进行重写

model存储模型类,重构了部分参数放入config.json

train_config是重构的


train.py和eval_cli.py使用lmdb进行重写,train.py现在会保存训练相关参数于checkpoints,可以配置自动覆盖

TensorBoard训练信息导出:




---
问题:
Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes,
but the current dtype in Gemma2ForCausalLM is torch.float32.
已解决

TODO:
考虑模型保存问题
解决```load_checkpoint```的可能冲突,争取直接使用checkpoint进行eval
考虑可视化问题