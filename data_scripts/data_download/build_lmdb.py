import os
import zipfile
import lmdb
import shutil
from tqdm import tqdm


def zip_to_lmdb(zip_path: str, lmdb_path: str, 
                batch_size: int = 10000, map_size_ratio: float = 2.0) -> str:
    """将zip转换为lmdb文件夹"""
    if not os.path.exists(zip_path):
        return f"⏭️ 跳过: {os.path.basename(zip_path)} 不存在"

    raw_size = sum(info.file_size for info in zipfile.ZipFile(zip_path, 'r').infolist() if not info.is_dir())
    map_size = int(raw_size * map_size_ratio)
    print(f"📊 原始大小: {raw_size/1024**3:.2f} GB | LMDB预分配: {map_size/1024**3:.2f} GB")

    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)
    os.makedirs(lmdb_path, exist_ok=True)

    #此处的max_readers为了防止锁表影响后续多线程访问
    env = lmdb.open(lmdb_path, map_size=map_size, subdir=True, readonly=False, max_readers=2048)
    txn = env.begin(write=True)
    count = 0
    pbar = None

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            files = [f for f in z.namelist() if not f.endswith('/')]
            total = len(files)
            pbar = tqdm(total=total, desc=f"📦 {os.path.basename(zip_path)}", unit="file")
            
            for fname in files:
                with z.open(fname) as f:
                    data = f.read()
                #修复嵌套压缩导致的文件前缀问题
                clean_name = fname.split('/')[-1] 
                key = clean_name.encode('utf-8')
                txn.put(key, data)
                count += 1
                pbar.update(1)


                if count % batch_size == 0:
                    txn.commit()
                    txn = env.begin(write=True)
    
        txn.commit()
        pbar.set_description_str(f"✅ {os.path.basename(zip_path)}", refresh=True)
        return f"✅ {os.path.basename(zip_path)} 完成"
    except Exception as e:
        if txn: txn.abort()
        if pbar: pbar.set_description_str(f"❌ 失败: {str(e)[:30]}", refresh=True)
        raise RuntimeError(f"写入 {lmdb_path} 失败") from e
    finally:
        if pbar: pbar.close()
        env.close()

if __name__ == "__main__":

#======配置区
    zip_root = "../../data/grasp_anything"
    lmdb_root = "../../data/grasp_anything/lmdb"
    zip_names = ["mask","image", "grasp_label_positive", "grasp_label_negative", 
                 "scene_description", ]
    map_size_ratios = [2.0, 2.0, 3.0, 3.0, 3.0]
    batch_size = 10000
    #便于单个使用
    skip_names = []
    #skip_names = ["mask", "image", "grasp_label_positive", "grasp_label_negative"]
#======注意batch_size

    tasks = []
    for name, ratio in zip(zip_names, map_size_ratios):
        if name in skip_names:
            continue

        zp = os.path.join(zip_root, f"{name}.zip")
        if os.path.exists(zp):
            tasks.append((zp, os.path.join(lmdb_root, name), batch_size, ratio))
        else:
            print(f"⏭️ 跳过: {name}.zip 不存在")


    print(f"🚀 开始单线程顺序转换，共 {len(tasks)} 个任务")
    for i, (zp, lp, bs, msr) in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] 正在处理: {os.path.basename(zp)}")
        zip_to_lmdb(zp, lp, bs, msr)
    print("\n🎉 全部转换任务执行完毕！")