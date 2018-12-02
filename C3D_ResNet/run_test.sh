
# base_folder=/media/qcxu/qcxuDisk/Dataset/scratch_dataset/videos/Video_11

folder=/media/qcxu/qcxuDisk/windows_datasets_all/videos_test/small_videos

for i in 1 2
do
    base_folder=$folder/3_$i
    echo "Hello, Welcome to $base_folder "

	# #### Generate n_frames files using utils/n_frames_ucf101_hmdb51.py
	# python utils/n_frames_ucf101_hmdb51.py ${base_folder}/C3D_clips


	#### Generate annotation file in json format similar to ActivityNet using utils/ucf101_json.py
	python 3D-ResNet/utils/scratch_json.py ${base_folder} testing


	# #### test
	python main_test_on_video.py --base_folder ${base_folder}

done
