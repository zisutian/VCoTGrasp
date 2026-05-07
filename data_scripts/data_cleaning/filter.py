import os
from tqdm import tqdm
import pandas as pd

import numpy as np
import torch
from torchvision.ops import box_iou
from ultralytics import YOLOWorld
from ultralytics import settings

import lmdb
import io
from PIL import Image

def mask_to_bbox_position(mask):
    """Get bbox position from boolean mask array."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return (x_min, y_min, x_max, y_max)  # (left, top, right, bottom)


#======配置区
LMDB_ROOT = "../../data/grasp_anything/lmdb"
IMAGE_LMDB = os.path.join(LMDB_ROOT, "image")
MASK_LMDB = os.path.join(LMDB_ROOT, "mask")

env_img = lmdb.open(IMAGE_LMDB, readonly=True, lock=False, readahead=False, meminit=False)
env_mask  = lmdb.open(MASK_LMDB,  readonly=True, lock=False, readahead=False, meminit=False)
txn_img = env_img.begin()
txn_mask = env_mask.begin()

device = "cuda"
settings.update({"runs_dir": "../../data/yolo_world_workspace/runs"})
model = YOLOWorld("../../data/yolo_world_workspace/yolov8s-worldv2.pt").to(device)  # or choose yolov8m/l-world.pt

load_path = "../../data/split/all.csv"
save_path = "../../data/split/all_filter.csv"
data = pd.read_csv(load_path)
iou_threshold = 0.25

class_batch_size = 100
filter_mask = []
for i in tqdm(range(data.shape[0])):
    data_id, obj_name, scene_description = data.iloc[i]
    image_id, obj_id = data_id.split("_")

    #mask = np.load(os.path.join(mask_root, f"{data_id}.npy"))
    mask_bytes = txn_mask.get(data_id.encode('utf-8'))
    mask = np.load(io.BytesIO(mask_bytes))

    bbox_label = mask_to_bbox_position(mask)
    bbox_label = torch.tensor(bbox_label).unsqueeze(0).to(device)

    if i % class_batch_size == 0:
        classes = data.iloc[i:i + class_batch_size, 1].tolist()
        classes = list(set(classes))
        # model.set_classes([obj_name])
        model.model.set_classes(classes, cache_clip_model=False) # see https://github.com/ultralytics/ultralytics/issues/20889
        model.model.names = classes
        if model.predictor:
            model.predictor.model.names = classes

    #results = model.predict(os.path.join(img_pil, f"{image_id}.jpg"), save=False, verbose=False)
    img_bytes = txn_img.get(image_id.encode('utf-8'))
    # 内存字节 → PIL Image (Ultralytics 原生支持，自动处理色彩空间与缩放)
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    results = model.predict(img_pil, save=False, verbose=False)

    results = results[0]
    bbox_pred = results.boxes.xyxy
    cls_pred = results.boxes.cls
    obj_cls = classes.index(obj_name)
    bbox_pred = bbox_pred[cls_pred == obj_cls]
    if bbox_pred.shape[0] != 1:
        filter_mask.append(False)
        continue
    iou = box_iou(bbox_pred, bbox_label).item()
    filter_mask.append(iou > iou_threshold)

#安全关闭lmdb环境
txn_img.abort()
txn_mask.abort()
env_img.close()
env_mask.close()


filterd_data = data[filter_mask]
filterd_data.to_csv(save_path, index=False, header=False)

print(f"before: {data.shape[0]}, after: {filterd_data.shape[0]}")