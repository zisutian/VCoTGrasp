export HF_ENDPOINT=https://hf-mirror.com
cd ../../data/grasp_anything

huggingface-cli download airvlab/Grasp-Anything --repo-type dataset --local-dir ./ --include "image_part_*"
huggingface-cli download airvlab/Grasp-Anything --repo-type dataset --local-dir ./ --include "grasp_label_positive.zip"
huggingface-cli download airvlab/Grasp-Anything --repo-type dataset --local-dir ./ --include "grasp_label_negative.zip"
huggingface-cli download airvlab/Grasp-Anything --repo-type dataset --local-dir ./ --include "scene_description.zip"
huggingface-cli download airvlab/Grasp-Anything --repo-type dataset --local-dir ./ --include "mask.zip"
cat image_part_aa image_part_ab > image.zip


# 对于mask的解压极其耗费时长,因此考虑使用lmdb进行存储和所有的运算
# 参考build_lmdb
# unzip -oq image.zip -d ./image 
# unzip -oq grasp_label_positive.zip -d ./grasp_label_positive 
# unzip -oq grasp_label_negative.zip -d ./grasp_label_negative 
# unzip -oq scene_description.zip -d ./scene_description 
# unzip -oq mask.zip -d ./mask 