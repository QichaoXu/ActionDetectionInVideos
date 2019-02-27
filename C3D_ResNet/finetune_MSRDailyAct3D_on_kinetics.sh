

base_folder=/media/qcxu/qcxuDisk/Dataset/MSRDailyAct3D

# ### Generate n_frames files using utils/n_frames_ucf101_hmdb51.py
#python utils/n_frames_ucf101_hmdb51.py ${base_folder}/Data

### Generate annotation file in json format similar to ActivityNet using utils/ucf101_json.py
#python utils/ucf101_json.py ${base_folder}/TrainTestList



python main.py --root_path ./data \
--video_path ${base_folder}/Data \
--annotation_path ${base_folder}/TrainTestList/ucf101_01.json \
--result_path results-MSRDailyAct3D-18 --dataset ucf101 --n_classes 400 --n_finetune_classes 16 \
--pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 4 --sample_duration 32 \
--model resnet --model_depth 18 --resnet_shortcut A --batch_size 16 --n_threads 4 --checkpoint 5