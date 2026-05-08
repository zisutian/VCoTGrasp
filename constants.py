root = "../VCoT-Grasp-self"

paligemma_model_id = "google/paligemma2-3b-mix-224"
pretrained_paligemma_dir = f"{root}/checkpoint/pretrained"

action_seq_len = 5
action_with_binned_angle_seq_len = 22
angle_bins = 18

split_root = f"{root}/data/grasp_anything/origin_split"
grasp_anything_planar_grasp_train_csv_path = f"{split_root}/train.csv"
grasp_anything_planar_grasp_test_seen_csv_path = f"{split_root}/test_seen.csv"
grasp_anything_planar_grasp_test_unseen_csv_path = f"{split_root}/test_unseen.csv"


grasp_dataset_root = f"{root}/data/grasp_anything/lmdb"
grasp_anything_rgb_root = f"{grasp_dataset_root}/image"
grasp_anything_mask_root = f"{grasp_dataset_root}/mask"
grasp_anything_description_root = f"{grasp_dataset_root}/scene_description"
grasp_anything_planar_grasp_root = f"{grasp_dataset_root}/grasp_label_positive"

real_image_root = "../mllm_data/grasp_real-world-v1/planar_data_process"
real_grasp_data_path = "../mllm_data/grasp_real-world-v1/grasp_real-world-v1_annotations_with_grasp_labels.xml"
real_bbox_data_path = "../mllm_data/grasp_real-world-v1/grasp_real-world-v1_annotations_with_object_detection.xml"
