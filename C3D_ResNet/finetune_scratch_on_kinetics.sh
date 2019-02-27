
base_folder=/media/qcxu/qcxuDisk/Dataset/scratch_dataset

# ## Generate n_frames files using utils/n_frames_ucf101_hmdb51.py
python utils/n_frames_ucf101_hmdb51.py ${base_folder}/hand_static_BG

# ## Generate annotation file in json format similar to ActivityNet using utils/ucf101_json.py
python utils/scratch_json.py ${base_folder}/TrainTestlist/hand_static_BG

### finetune from UCF101 and iterate over 10 train list
python main_train_on_video.py --root_path ./data \
--video_path ${base_folder}/hand_static_BG \
--annotation_path ${base_folder}/TrainTestlist/hand_static_BG/ucf101_01.json \
--result_path results-scratch-18-static_BG-30-iter6 --sample_duration 30 --dataset ucf101 --n_classes 400 --n_finetune_classes 3 \
--pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 3 --n_epochs 300 \
--model resnet --model_depth 18 --resnet_shortcut A --batch_size 25 --n_threads 1 --checkpoint 20


# #### finetune from UCF101
# python main_train_on_video.py --root_path ./data \
# --video_path ${base_folder}/hand_static_BG \
# --annotation_path ${base_folder}/TrainTestlist/hand_static_BG/ucf101_01.json \
# --result_path results-scratch-18-static_BG-30 --sample_duration 30 --dataset ucf101 --n_classes 400 --n_finetune_classes 3 \
# --pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 3 \
# --model resnet --model_depth 18 --resnet_shortcut A --batch_size 25 --n_threads 1 --checkpoint 20


# # #### finetune from old scratch-dataset
# python main_train_on_video.py --root_path ./data \
# --video_path ${base_folder}/hand_static_BG \
# --annotation_path ${base_folder}/TrainTestlist/hand_static_BG/ucf101_01.json \
# --result_path results-scratch-18-static_BG-30-finetuned --sample_duration 30 --dataset ucf101 --n_classes 3 --n_finetune_classes 3 \
# --pretrain_path results-scratch-18-static_BG-30/save_200.pth --ft_begin_index 3 \
# --model resnet --model_depth 18 --resnet_shortcut A --batch_size 25 --n_threads 1 --checkpoint 20

# #### tempory test
# python main_train_on_video.py --root_path ./data \
# --video_path ${base_folder}/hand_static_BG \
# --annotation_path ${base_folder}/TrainTestlist/hand_static_BG/ucf101_01.json \
# --result_path results-scratch-18-static_BG-30-test --sample_duration 30 --dataset ucf101 --n_classes 400 --n_finetune_classes 3 \
# --pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 3 --n_epochs 1 \
# --model resnet --model_depth 18 --resnet_shortcut A --batch_size 1 --n_threads 1 --checkpoint 20
