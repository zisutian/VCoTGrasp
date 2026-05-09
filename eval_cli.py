import argparse
import io
import json
import os

import numpy as np
import torch
from PIL import ImageDraw
from tqdm import tqdm

import data
from constants import *
from inference import VCoTGraspInferencer, eval_grasp_all_labels


MODEL_CONFIG_NAME = "config.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_eval_config(checkpoint_dir):
    checkpoint_config_path = os.path.join(checkpoint_dir, MODEL_CONFIG_NAME)
    if os.path.isfile(checkpoint_config_path):
        model_config = load_json(checkpoint_config_path)
        arch_config = model_config.get("arch_config", {})
        return {
            "use_bbox": arch_config.get("use_bbox"),
            "action_head": arch_config.get("action_head"),
            "source": checkpoint_config_path,
        }

    raise FileNotFoundError(
        f"Cannot find {MODEL_CONFIG_NAME} in checkpoint dir: {checkpoint_dir}"
    )


def resolve_eval_config(args):
    checkpoint_config = load_checkpoint_eval_config(args.load_checkpoint_dir)
    use_bbox = checkpoint_config["use_bbox"]
    action_head = checkpoint_config["action_head"]

    if use_bbox is None:
        raise ValueError("use_bbox is missing. Add it to checkpoint config.json.")
    if not action_head:
        raise ValueError("action_head is missing. Add it to checkpoint config.json.")

    return argparse.Namespace(
        load_checkpoint_dir=args.load_checkpoint_dir,
        test_split=args.test_split,
        use_bbox=use_bbox,
        action_head=action_head,
        device=args.device,
        visualize_dir=args.visualize_dir,
        result_dir=args.result_dir,
        use_lora=args.use_lora,
        config_source=checkpoint_config["source"],
        save_image_range=args.save_image_range,
        iou_threshold=args.iou_threshold,
        angle_threshold=args.angle_threshold,
    )


def get_split_csv_path(test_split):
    if test_split == "seen":
        return grasp_anything_planar_grasp_test_seen_csv_path
    if test_split == "unseen":
        return grasp_anything_planar_grasp_test_unseen_csv_path
    raise ValueError(f"Invalid test split: {test_split}")


def eval_split(args, test_split):
    inferencer = VCoTGraspInferencer(
        args.load_checkpoint_dir,
        device=args.device,
        use_lora=args.use_lora,
    )

    visualize_dir = os.path.join(args.visualize_dir, test_split)
    os.makedirs(visualize_dir, exist_ok=True)

    ds = data.GraspAnythingForGraspGeneration(
        get_split_csv_path(test_split),
        grasp_anything_rgb_root,
        grasp_anything_planar_grasp_root,
        grasp_anything_mask_root,
        use_bbox=args.use_bbox,
        action_head_type=args.action_head,
    )

    n_success = 0
    n_valid = 0
    n_all = len(ds)
    for i in tqdm(range(len(ds))):
        image, prompt, _, _, obj_name, _ = ds[i]
        grasp_id, _, _ = ds.data.iloc[i]

        mask_bytes = data._get_lmdb_bytes(ds.env_mask, f"{grasp_id}.npy")
        mask = np.load(io.BytesIO(mask_bytes))
        bbox_gt = data.mask_to_bbox_position(mask)

        grasp_bytes = data._get_lmdb_bytes(ds.env_grasp, f"{grasp_id}.pt")
        grasp_gts_tensor = torch.load(io.BytesIO(grasp_bytes), map_location="cpu", weights_only=False)
        grasp_gts_list = grasp_gts_tensor[:, 1:].tolist()

        grasp_pred, bbox_pred = inferencer.generate_postprocess(image, prompt, obj_name)
        if grasp_pred is None or (args.use_bbox and bbox_pred is None):
            continue

        success = eval_grasp_all_labels(grasp_pred, grasp_gts_list, args.iou_threshold, args.angle_threshold)
        n_success += int(success)
        n_valid += 1

        if i < args.save_image_range:
            if args.use_bbox:
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox_pred, outline="yellow", width=3)
                draw.rectangle(bbox_gt, outline="blue", width=3)
            for grasp_gt in grasp_gts_list:
                image = data.draw_grasp_rectangle(image, grasp_gt, "green")
            image = data.draw_grasp_rectangle(image, grasp_pred, "red")
            save_path = os.path.join(visualize_dir, f"{i:02d}_{success}_{obj_name}.jpg")
            image.save(save_path)

    res = (
        f"checkpoint: {args.load_checkpoint_dir}\n"
        f"config source: {args.config_source}\n"
        f"test split: {test_split}\n"
        f"use bbox: {args.use_bbox}\n"
        f"action head: {args.action_head}\n"
        f"num success: {n_success}, num valid: {n_valid}, num all: {n_all}\n"
        f"success rate: {n_success / n_all * 100} %\n"
    )

    os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, f"{test_split}.txt")
    with open(result_path, "w", encoding="utf-8") as file:
        file.write(res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-checkpoint-dir", type=str, required=True, help="finished checkpoint dir")
    parser.add_argument("--test-split", type=str, default="all", choices=["all", "seen", "unseen"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize-dir", type=str, default="results/visualize/")
    parser.add_argument("--result-dir", type=str, default="results/")
    parser.add_argument("--use-lora", action="store_true")

    parser.add_argument("--save-image-range", type=int, default=100)
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    parser.add_argument("--angle-threshold", type=float, default=30)

    args = resolve_eval_config(parser.parse_args())
    print(f"Loaded eval config from {args.config_source}")
    print(f"use_bbox={args.use_bbox}, action_head={args.action_head}")

    if args.test_split == "all":
        eval_split(args, "seen")
        eval_split(args, "unseen")
    else:
        eval_split(args, args.test_split)


if __name__ == "__main__":
    main()
