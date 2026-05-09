# VCoTGrasp-self-v2

checkpoint内存放外部下载的模型:
- pretrained: PaliGemma2预训练模型.
- vcot: 可选,老版训练好的vcot模型,可用于load_checkpoint.

checkpoints存放本项目训练产生的checkpoint.

data存放数据集,origin_split是原作者处理后的csv文件.

data_scripts是数据集下载与处理脚本,当前数据读取已改为lmdb.

model存放模型类.

train存放训练参数和训练脚本.

eval存放评估脚本.

## Checkpoint配置规则

checkpoint/config.json是模型结构的唯一来源.

以下参数属于模型结构,必须以checkpoint/config.json为准:
- arch_config.use_bbox
- arch_config.action_head
- 会改变输入格式,token,head结构,输出后处理的其他参数

checkpoint/train_config.json只保存训练过程参数,例如:
- lr
- batch size
- scheduler
- train_dataset
- data_ratio
- bbox_ratio
- train_image_encoder
- train_image_projector
- train_embeddings
- train_lm

路径和运行名称不会写入checkpoint/train_config.json:
- load_checkpoint_dir
- model_config_path
- save_root
- tensorboard_root
- train_config_path
- run_name
- overwrite_checkpoints

## 训练

新训练时,train json可以配置use_bbox/action_head.

train.py会把这两个字段写入模型config,并保存到checkpoint/config.json.

resume时,checkpoint中已有的配置无法通过当前train json更改.

模型架构始终以load_checkpoint_dir/config.json为准.

如果checkpoint包含train_config.json,训练过程参数也以checkpoint/train_config.json为准.

当前train json中的冲突字段只会触发warning,不会覆盖checkpoint中的值.

老checkpoint例如checkpoint/vcot可能没有train_config.json. 此时训练过程参数使用当前train json,但模型架构仍然从checkpoint/vcot/config.json读取.

train/resume.json用于新checkpoint继续训练,通常只需要指定:
- load_checkpoint_dir
- save_root
- tensorboard_root
- run_name

train/resume_old.json用于老checkpoint继续训练,需要写完整训练过程参数.

## 评估

eval_cli.py的模型架构信息只从checkpoint/config.json读取.

评估脚本不再显式传入use_bbox/action_head.

## 已知问题

Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes,
but the current dtype in Gemma2ForCausalLM is torch.float32.

已解决.

## TODO

- 考虑可视化问题.
