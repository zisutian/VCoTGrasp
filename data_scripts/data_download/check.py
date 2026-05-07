# import zipfile
# from collections import Counter
# import os

# with zipfile.ZipFile("../../data/grasp_anything/mask.zip") as z:
#     files = [i for i in z.infolist() if not i.filename.endswith('/')]
#     total_files = len(files)
#     total_uncompressed = sum(i.file_size for i in files)
#     total_compressed = sum(i.compress_size for i in files)
    
# print(f"📦 mask.zip 统计:")
# print(f"  文件数量:        {total_files:,}")
# print(f"  解压后总大小:    {total_uncompressed/1024**3:.2f} GB")
# print(f"  压缩后总大小:    {total_compressed/1024**3:.2f} GB")
# print(f"  平均压缩率:      {total_compressed/total_uncompressed*100:.3f}%")


import lmdb
import os
LMDB_ROOT = "../../data/grasp_anything/lmdb"

LABEL_LMDB = os.path.join(LMDB_ROOT, "grasp_label_positive")
env_label = lmdb.open(LABEL_LMDB, readonly=True, lock=False, readahead=False, meminit=False)
print("🔍 真实 Key 样本：")
with env_label.begin() as txn:
    for k, _ in txn.cursor().iternext():
        print(f"{k.decode('utf-8')}")