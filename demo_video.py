
from detection import detection

import os
import sys
import cv2


if __name__ == "__main__":

	T = 45
	reg_model_file = 'results-scratch-18/save_200.pth'
	detection = detection(reg_model_file, is_static_BG=False, thres=0.9)

	video_path = '/media/qcxu/qcxuDisk/windows_datasets_all/videos_test/small_videos/1_7.avi'
	stream = cv2.VideoCapture(video_path)
	grabbed = True
	frame_id = 0
	img_out_all = None
	imglist = []
	while grabbed:
		(grabbed, frame) = stream.read()
		frame_id += 1

		if frame_id <= T:
			imglist.append(frame)
			if frame_id == T:
				frame_id = 0
				img_out_all = detection.run(imglist)
				imglist = []
			else:
				if img_out_all is None:
					continue
				else:
					cv2.imshow('skeletons', img_out_all[frame_id-1])
					cv2.waitKey(15)

	detection.print_runtime()


	# base_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/others/clips/Video_11_1_1'
	# imglist = []
	# for img_name in os.listdir(base_folder):
	#	 if img_name.endswith('jpg'):
	#		 imglist.append(cv2.imread(os.path.join(base_folder, img_name)))

	# img_out_all = detection.run(imglist)

