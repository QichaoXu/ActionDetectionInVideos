
base_folder=/media/qcxu/qcxuDisk/Dataset/scratch_dataset/videos/Video_11

#### Generate n_frames files using utils/n_frames_ucf101_hmdb51.py
python utils/n_frames_ucf101_hmdb51.py ${base_folder}/C3D_clips


#### Generate annotation file in json format similar to ActivityNet using utils/ucf101_json.py
python utils/scratch_json.py ${base_folder}


# #### test
python main_test_on_video.py