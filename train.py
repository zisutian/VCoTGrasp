import os
import argparse
import json
import shutil
import glob
import warnings

import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import get_cosine_schedule_with_warmup, get_constant_schedule, set_seed
from accelerate import Accelerator

from model import VCoTGraspConfig, VCoTGraspForConditionalGeneration, VCoTGraspProcessor
from data import get_dataloaders


DEFAULT_TRAIN_CONFIG_PATH = "train/grasp_anything_mlp.json"
TRAIN_CONFIG_NAME = "train_config.json"
CHECKPOINT_TRAIN_CONFIG_EXCLUDE_KEYS = {
    "load_checkpoint_dir",
    "model_config_path",
    "save_root",
    "tensorboard_root",
    "train_config_path",
    "run_name",
    "overwrite_checkpoints",
    "torch_dtype",
    "use_bbox",
    "action_head",
}


def load_train_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["train_config_path"] = config_path
    return argparse.Namespace(**config)


def apply_checkpoint_train_config(args):
    if not args.load_checkpoint_dir:
        return args

    checkpoint_train_config_path = os.path.join(args.load_checkpoint_dir, TRAIN_CONFIG_NAME)
    if not os.path.isfile(checkpoint_train_config_path):
        warnings.warn(
            f"Cannot find {TRAIN_CONFIG_NAME} in checkpoint dir: {args.load_checkpoint_dir}. "
            "Using the current train config.",
            UserWarning,
        )
        return args

    with open(checkpoint_train_config_path, "r", encoding="utf-8") as f:
        checkpoint_train_config = json.load(f)

    current_config = vars(args)
    conflicts = []
    for key, value in checkpoint_train_config.items():
        if key in CHECKPOINT_TRAIN_CONFIG_EXCLUDE_KEYS:
            continue
        if key in current_config and current_config[key] != value:
            conflicts.append((key, current_config[key], value))

    if conflicts:
        conflict_lines = [
            f"  {key}: current={current_value!r}, checkpoint={checkpoint_value!r}"
            for key, current_value, checkpoint_value in conflicts
        ]
        warnings.warn(
            "Current train config conflicts with checkpoint train config:\n"
            + "\n".join(conflict_lines)
            + "\nUsing the checkpoint's training settings for these fields.",
            UserWarning,
        )

    for key, value in checkpoint_train_config.items():
        if key not in CHECKPOINT_TRAIN_CONFIG_EXCLUDE_KEYS:
            current_config[key] = value

    return argparse.Namespace(**current_config)


def warn_if_train_config_arch_conflicts(args, arch_config, source):
    conflicts = []
    for key in ("use_bbox", "action_head"):
        if hasattr(args, key) and getattr(args, key) != getattr(arch_config, key):
            conflicts.append((key, getattr(args, key), getattr(arch_config, key)))

    if conflicts:
        conflict_lines = [
            f"  {key}: train_config={train_value!r}, {source}={arch_value!r}"
            for key, train_value, arch_value in conflicts
        ]
        warnings.warn(
            "Train config architecture choices conflict with checkpoint config:\n"
            + "\n".join(conflict_lines)
            + f"\nUsing architecture choices from {source}.",
            UserWarning,
        )


def save_checkpoint_with_train_config(model, save_dir, train_args, overwrite_pattern=None):
    if getattr(train_args, "overwrite_checkpoints", False):
        if overwrite_pattern is not None:
            for old_dir in glob.glob(os.path.join(os.path.dirname(save_dir), overwrite_pattern)):
                if os.path.isdir(old_dir):
                    shutil.rmtree(old_dir)
        elif os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
    model.save_pretrained(save_dir)
    train_config = {
        key: value
        for key, value in vars(train_args).items()
        if key not in CHECKPOINT_TRAIN_CONFIG_EXCLUDE_KEYS
    }
    with open(os.path.join(save_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)


def main(args):
    set_seed(args.seed)
    accelerator = Accelerator(log_with="tensorboard", project_dir=args.tensorboard_root)

    if not args.load_checkpoint_dir:
        # init a model
        model_config = VCoTGraspConfig.from_json_file(args.model_config_path)
        if hasattr(args, "use_bbox"):
            model_config.arch_config.use_bbox = args.use_bbox
        if hasattr(args, "action_head"):
            model_config.arch_config.action_head = args.action_head
        model_config.set_attn_implementation(**args.attn_implementation)
        model = VCoTGraspForConditionalGeneration(model_config)
    else:
        model = VCoTGraspForConditionalGeneration.from_pretrained(
            args.load_checkpoint_dir,
            attn_implementation=args.attn_implementation,
        )
        warn_if_train_config_arch_conflicts(args, model.config.arch_config, "checkpoint config")

    runtime_arch_config = model.config.arch_config
    processor = VCoTGraspProcessor(runtime_arch_config)

    hyper_params = vars(args).copy()
    hyper_params["use_bbox"] = runtime_arch_config.use_bbox
    hyper_params["action_head"] = runtime_arch_config.action_head
    hyper_params.update(
        num_processes=accelerator.num_processes,
        gradient_accumulation_steps=accelerator.gradient_accumulation_steps,
    )

    model.set_trainable(
        image_encoder=args.train_image_encoder,
        image_projector=args.train_image_projector,
        embeddings=args.train_embeddings,
        lm=args.train_lm,
        action_head=True,
    )
    train_dataloader, eval_dataloader, _ = get_dataloaders(
        runtime_arch_config,
        args.train_dataset,
        processor,
        args.train_batch_size_per_gpu,
        args.eval_batch_size_per_gpu,
        bbox_ratio=args.bbox_ratio,
        data_ratio=args.data_ratio,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # on multiple gpus, lr scheduler will step multiple times after one gradient update, so multiply steps by num_processes, see https://github.com/huggingface/accelerate/issues/2142
    num_training_steps = (len(train_dataloader) * args.train_epoch) // accelerator.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    if args.lr_scheduler == "cosine_schedule_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    elif args.lr_scheduler == "cosine_schedule":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=5e-6)
    elif args.lr_scheduler == "constant_schedule":
        scheduler = get_constant_schedule(optimizer)
    else:
        raise NotImplementedError("Invalid scheduler")
    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )

    accelerator.init_trackers(args.run_name, config=hyper_params)
    save_dir = os.path.join(args.save_root, args.run_name)
    start_epoch = 0
    global_step = 0

    for epoch in range(start_epoch, args.train_epoch):
        model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                loss_info = outputs.loss_info
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                global_step += 1
                loss = accelerator.reduce(loss, "mean")  # per sample loss
                loss_info = "".join([f" {k}: {v:.5f} " for k, v in loss_info.items()])
                accelerator.log({"epoch": epoch, "loss": loss.item()}, step=global_step)
                accelerator.print(f"epoch: {epoch}  global_step: {global_step}  loss: {loss.item():.5f} " + loss_info)

                # eval, must check in "if accelerator.sync_gradients"
                if global_step % args.eval_every_n_steps == 0:
                    model.eval()
                    losses = []
                    for batch in eval_dataloader:
                        with torch.no_grad():
                            outputs = model(**batch, use_cache=False)
                        loss = outputs.loss
                        losses.append(accelerator.gather_for_metrics(loss.repeat(args.eval_batch_size_per_gpu)))

                    losses = torch.cat(losses)
                    eval_loss = torch.mean(losses)
                    accelerator.log({"epoch": epoch, "eval_loss": eval_loss.item()}, step=global_step)
                    accelerator.print(f"epoch: {epoch}, global_step: {global_step}, eval_loss: {eval_loss.item()}")
                    model.train()

                if args.save_every_n_steps > 0 and global_step % args.save_every_n_steps == 0:
                    version_dir = os.path.join(save_dir, f"epoch{epoch}_step{global_step}_latest")
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        save_checkpoint_with_train_config(
                            unwrapped_model,
                            version_dir,
                            args,
                            overwrite_pattern=f"epoch{epoch}_step*_latest",
                        )

        version_dir = os.path.join(save_dir, f"epoch{epoch}_step{global_step}_final")
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            save_checkpoint_with_train_config(
                unwrapped_model,
                version_dir,
                args,
            )

    accelerator.print("Training end")
    accelerator.end_training()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", type=str, default=DEFAULT_TRAIN_CONFIG_PATH, help="training config json path")

    cli_args = parser.parse_args()
    args = load_train_config(cli_args.train_config)
    args = apply_checkpoint_train_config(args)
    main(args)
