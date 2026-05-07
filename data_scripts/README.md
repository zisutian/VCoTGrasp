pip install lmdb

原有下载代码对于mask的解压极其耗费时长,因此使用lmdb进行存储和所有的运算
下载后使用build_lmdb.py进行lmdb的构建
注意mask.zip的大小,原始大小为300GB左右,全部数据大概为1T多


make_index_planar_grasp.py用于筛选低质量数据,并生成all.csv文件
随后使用filter.py筛选,生成all_fliter.csv文件
最后使用split.py,该文件用于划分数据

filter.py需要yolo_world的环境,因此有对应配置环境文件