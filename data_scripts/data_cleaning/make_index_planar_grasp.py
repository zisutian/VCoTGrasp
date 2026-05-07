"""
filter data that:
have a obj name "nan"
have too few mask points
bbox don't match the obj name
LMDB适配
"""

import os
import csv
import numpy as np
import pickle
from tqdm import tqdm

import lmdb
import io

def merge_names(s, names):
    """
    deal with data like:
    ('A green lime, a brown wooden cutting board, and a silver knife with a black handle', ['lime', 'cutting', 'board', 'knife'])
    merge names into: ['lime', 'cutting board', 'knife']
    """
    merged = []
    i = 0
    while i < len(names) - 1:
        cur = f"{names[i]} {names[i+1]}"
        if cur in s:
            merged.append(cur)
            i += 2
        else:
            merged.append(names[i])
            i += 1

    if i == len(names) - 1:
        merged.append(names[-1])

    return merged

#======配置区
LMDB_ROOT = "../../data/grasp_anything/lmdb"
CSV_PATH = "../../data/grasp_anything/split/all.csv"

LABEL_LMDB = os.path.join(LMDB_ROOT, "grasp_label_positive")
DESC_LMDB = os.path.join(LMDB_ROOT, "scene_description")
MASK_LMDB = os.path.join(LMDB_ROOT, "mask")

MASK_POINT_THRESH = 400
# csv_path = "split/grasp_anything_filter/all.csv"
# ds_root = "/baishuanghao/mllm_data/grasp_anything"
# image_root = os.path.join(ds_root, "image")
# label_root = os.path.join(ds_root, "grasp_label_positive")
# description_root = os.path.join(ds_root, "scene_description")
# mask_root = os.path.join(ds_root, "mask")
# mask_point_thresh = 400

env_label = lmdb.open(LABEL_LMDB, readonly=True, lock=False, readahead=False, meminit=False)
env_desc  = lmdb.open(DESC_LMDB,  readonly=True, lock=False, readahead=False, meminit=False)
env_mask  = lmdb.open(MASK_LMDB,  readonly=True, lock=False, readahead=False, meminit=False)

try:
    with env_label.begin() as txn:
        cursor = txn.cursor()
        data_ids = []
        for key in cursor.iternext(values=False):
            k_str = key.decode('utf-8')
            if k_str.endswith('.pt'):
                data_ids.append(k_str.removesuffix('.pt'))
        data_ids.sort()
    print(f"📊 从 LMDB 加载到 {len(data_ids)} 个有效 data_id")
# data_ids = os.listdir(label_root)
# data_ids = [f.removesuffix(".pt") for f in data_ids if f.endswith(".pt")]
# data_ids.sort()

    data = []
    skip_nan = 0
    skip_not_enough_points = 0
    skip_index = 0
    with env_desc.begin() as txn_desc, env_mask.begin() as txn_mask:
        with tqdm(data_ids, desc="过滤处理", unit="id") as pbar:
            #拆分scene_id和obj_index
            #for data_id in tqdm(data_ids):
            #   scene_id, obj_index = data_id.split("_")
            #   obj_index = int(obj_index)
            for data_id in pbar:
                parts = data_id.rsplit('_', 1)
                if len(parts) != 2:
                    continue
                scene_id, obj_idx_str = parts
                try:
                    obj_index = int(obj_idx_str)
                except ValueError:
                    continue

                #读取场景描述
                #description_path = os.path.join(description_root, f"{scene_id}.pkl")
                #with open(description_path, "rb") as f:
                #   description_data = pickle.load(f)
                desc_key = f"{scene_id}.pkl".encode('utf-8')
                desc_bytes = txn_desc.get(desc_key)
                if desc_bytes is None:
                    continue
                description_data = pickle.loads(desc_bytes)

                description = description_data[0]
                obj_names = description_data[1]
                obj_names = merge_names(description, obj_names)
                if obj_index >= len(obj_names):
                    skip_index += 1
                    continue
                obj_name = obj_names[obj_index]
                if obj_name in ["", "nan", "NaN"]:
                    skip_nan += 1
                    continue
                    
                #读取mask
                # mask_path = os.path.join(mask_root, f"{data_id}.npy")
                # mask = np.load(mask_path)
                mask_key = f"{data_id}.npy".encode('utf-8')
                mask_bytes = txn_mask.get(mask_key)
                if mask_bytes is None:
                    continue
                # np.load 直接读取内存字节流
                mask = np.load(io.BytesIO(mask_bytes))

                mask_points = np.count_nonzero(mask)
                if mask_points < MASK_POINT_THRESH:
                    skip_not_enough_points += 1
                    continue

                data.append([data_id, obj_name, description])

    #写入csv
    # with open(csv_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(data)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"\n过滤完成, 保留 {len(data)} 条数据")
    print(f"skip nan:                {skip_nan}")
    print(f"skip index:              {skip_index}")
    print(f"skip not enough points:  {skip_not_enough_points}")

finally:
    env_label.close()
    env_desc.close()
    env_mask.close()