

base_folder=/media/qcxu/qcxuDisk/Dataset/UCF

### Generate n_frames files using utils/n_frames_ucf101_hmdb51.py
python utils/n_frames_ucf101_hmdb51.py ${base_folder}/UCF-101-img-folder

### Generate annotation file in json format similar to ActivityNet using utils/ucf101_json.py
python utils/ucf101_json.py ${base_folder}/ucfTrainTestlist



python main.py --root_path ./data \
--video_path ${base_folder}/UCF-101-img-folder \
--annotation_path ${base_folder}/ucfTrainTestlist/ucf101_01.json \
--result_path results-ucf-resnet-18-16 --dataset ucf101 --n_classes 400 --n_finetune_classes 101 \
--pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 4 \
--model resnet --model_depth 18 --resnet_shortcut A --batch_size 128 --n_threads 4 --checkpoint 5