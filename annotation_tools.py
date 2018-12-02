import numpy as np
import os
import cv2
from shutil import copyfile

from skimage.transform import resize
from skimage.io import imsave
import imageio


class annotation_tools:
	def __init__(self):
		None

	def __read_annos(self, anno_name):
		file = open(anno_name, 'r')
		annos = []
		line_id = 0
		for line in file:
			line_id += 1
			line = line[:-1].split('\t')
			if line_id > 3 and len(line) == 6:
				annos += [[int(line[0]), int(line[1]), int(float(line[2])), \
				int(float(line[3])), int(float(line[4])), int(float(line[5]))]]

		annos = np.array(annos)
		num_labels = annos[-1][1]
		annos_res = []
		for lable_id in range(1, num_labels+1):
			xxx = annos[annos[:, 1] == lable_id]
			if xxx.shape[0] != 0:
				annos_res.append(xxx)

		return annos_res

	def __get_ROI_value(self, img, anno):
		width = img.shape[0]
		height = img.shape[1]
		xmin = max(0, anno[2])
		ymin = max(0, anno[3])
		xmax = min(height, anno[2]+anno[4])
		ymax = min(width, anno[3]+anno[5])
		return xmin, xmax, ymin, ymax

	def __plot_anno(self, img, anno):
		xmin, xmax, ymin, ymax = self.__get_ROI_value(img, anno)
		cv2.rectangle(img, (xmin,ymin), (xmax,ymax),(0,0,255),3,4,0)
		return img

	def __get_ROI_from_anno(self, img, anno):
		xmin, xmax, ymin, ymax = self.__get_ROI_value(img, anno)
		return img[ymin:ymax, xmin:xmax, :]

	def vis_anno_result(self, video_name, anno_name):
		if not video_name.lower().endswith(('.avi', '.mp4', '.mkv', '.mpg', '.m4v', 'wmv')):
			print('**** Bad video format ****', video_name)
			return
		if not os.path.isfile(video_name):
			print('**** Video not exists ****', video_name)
			return
		if not os.path.isfile(anno_name):
			print('**** Annotation file not exists ****')
			return
		print(anno_name)

		annos_all = self.__read_annos(anno_name)
		for annos in annos_all:

			frame_id = 0
			start_frame_id = annos[0][0]
			end_frame_id = annos[-1][0]

			cap = cv2.VideoCapture(video_name)
			cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
			while frame_id+start_frame_id < end_frame_id:
				try:
					ret, img = cap.read()
				except:
					print('**** No frame_id in this video ****')
					break
				anno = annos[frame_id]
				frame_id += 1
				img = self.__plot_anno(img, anno)
				cv2.imshow('Label Result', img)
				cv2.waitKey(15)

	def get_clip_from_annos(self, video_name, anno_name, save_folder=None, n_samples_for_each_video=45):
		if not video_name.lower().endswith(('.avi', '.mp4', '.mkv', '.mpg', '.m4v', 'wmv')):
			print('**** Bad video format ****', video_name)
			return
		if not os.path.isfile(video_name):
			print('**** Video not exists ****', video_name)
			return
		if not os.path.isfile(anno_name):
			print('**** Annotation file not exists ****')
			return
		print(anno_name)

		annos_all = self.__read_annos(anno_name)
		for annos in annos_all:
			frame_id = 0
			start_frame_id = annos[0][0]
			end_frame_id = annos[-1][0]

			num_videos = (end_frame_id - start_frame_id) // n_samples_for_each_video
			print(num_videos)

			cap = cv2.VideoCapture(video_name)
			cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
			while frame_id+start_frame_id < end_frame_id and frame_id < num_videos*n_samples_for_each_video:
				ret, img = cap.read()
				if not ret:
					print('**** No frame_id in this video ****', frame_id+start_frame_id)
					break
				anno = annos[frame_id]
				frame_id += 1
				img = self.__get_ROI_from_anno(img, anno)
				cv2.imshow('Label Result', img)
				cv2.waitKey(15)

				if save_folder is not None:
					out_video_id = (frame_id - 1) // n_samples_for_each_video + 1
					save_folder_tmp = save_folder + '_' + str(annos[0][1]) + '_' + str(out_video_id)
					if not os.path.exists(save_folder_tmp):
						os.makedirs(save_folder_tmp)
					img_name = 'image_{:0>5d}.jpg'.format(frame_id % (n_samples_for_each_video))
					cv2.imwrite(save_folder_tmp+'/'+img_name, img)

	def __get_annos_intervals_from_annos(self, anno_name):
		if not os.path.isfile(anno_name):
			return None

		file = open(anno_name, 'r')
		annos = []
		line_id = 0
		for line in file:
			line_id += 1
			line = line[:-1].split('\t')
			if line_id > 3 and len(line) == 6:
				annos += [[int(line[0]), int(line[1])]]

		annos = np.array(annos)
		num_labels = annos[-1][1]
		annos_intervals = []
		for lable_id in range(1, num_labels+1):
			xxx = annos[annos[:, 1] == lable_id]
			if xxx.shape[0] != 0:
				annos_intervals.append([lable_id, xxx[0][0], xxx[-1][0]])

		return annos_intervals

	def __get_others_frame_from_annos(self, anno_name1, anno_name2, num_frames, n_samples_for_each_video, min_gap=20):
		annos_ins1 = self.__get_annos_intervals_from_annos(anno_name1)
		annos_ins2 = self.__get_annos_intervals_from_annos(anno_name2)
		if annos_ins1 is None and annos_ins2 is not None:
			annos_ins = annos_ins2
		if annos_ins1 is not None and annos_ins2 is None:
			annos_ins = annos_ins1
		if annos_ins1 is None and annos_ins2 is None:
			annos_ins = []
		if annos_ins1 is not None and annos_ins2 is not None:
			annos_ins = annos_ins1
			for annos_in in annos_ins2:
				annos_ins.append(annos_in)

		annos_ins.append([-1, num_frames, -1])

		spare_ins = []
		pre_frame_id = 1
		for annos_in in annos_ins:
			start_frame_id = max(1, annos_in[1] - min_gap)
			end_frame_id = min(num_frames, annos_in[2] + min_gap)
			if start_frame_id - pre_frame_id > n_samples_for_each_video:
				spare_ins.append([len(spare_ins)+1, pre_frame_id, start_frame_id])
			pre_frame_id = end_frame_id

		return spare_ins

	def get_others_class_from_annos(self, video_name, anno_name1, anno_name2, save_folder=None, n_samples_for_each_video=45):
		if not video_name.lower().endswith(('.avi', '.mp4', '.mkv', '.mpg', '.m4v', 'wmv')):
			print('**** Bad video format ****', video_name)
			return
		if not os.path.isfile(video_name):
			print('**** Video not exists ****', video_name)
			return

		print(anno_name1, anno_name2)

		cap = cv2.VideoCapture(video_name)
		num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		annos_all = self.__get_others_frame_from_annos(anno_name1, anno_name2, num_frames, n_samples_for_each_video)
		for annos in annos_all:
			frame_id = 0
			start_frame_id = annos[1]
			end_frame_id = annos[2]

			num_videos = (end_frame_id - start_frame_id) // n_samples_for_each_video
			print(num_videos)

			cap = cv2.VideoCapture(video_name)
			cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
			while frame_id+start_frame_id < end_frame_id and frame_id < num_videos*n_samples_for_each_video:
				ret, img = cap.read()
				if not ret:
					print('**** No frame_id in this video ****', frame_id+start_frame_id)
					break
				frame_id += 1
				cv2.imshow('Label Result', img)
				cv2.waitKey(15)

				if save_folder is not None:
					out_video_id = (frame_id - 1) // n_samples_for_each_video + 1
					save_folder_tmp = save_folder + '_' + str(annos[0]) + '_' + str(out_video_id)
					if not os.path.exists(save_folder_tmp):
						os.makedirs(save_folder_tmp)
					img_name = 'image_{:0>5d}.jpg'.format(frame_id % (n_samples_for_each_video))
					cv2.imwrite(save_folder_tmp+'/'+img_name, img)

	### for windows_dataset
	def get_TrainTestlist(self, clip_folder, Trainlist_name, Testlist_name):
		label_map = {'clean':'1', 'normal':'2', 'pick':'3', 'scratch':'4'}

		Trainlist = open(Trainlist_name, 'w')
		Testlist = open(Testlist_name, 'w')
		count = 0
		for sub in sorted(os.listdir(clip_folder)):
			sub_folder = clip_folder + '/' + sub
			for subsub in sorted(os.listdir(sub_folder)):
				contents = sub + '/' + subsub + ' ' + label_map[sub] + '\n'
				if count % 4 == 0: # test
					Testlist.write(contents)
				else:
					Trainlist.write(contents)
				count += 1

		Trainlist.close()
		Testlist.close()

	### for windows_dataset
	def split_into_n_samples(self, in_folder, out_folder, n_samples_for_each_video=45, min_n_samples=40):
		if not os.path.exists(out_folder):
			os.makedirs(out_folder)

		for sub in os.listdir(in_folder):
			sub_in_folder = os.path.join(in_folder, sub)
			image_list = [img for img in os.listdir(sub_in_folder) if img.endswith('.jpg')]
			n_folders = len(image_list) // n_samples_for_each_video
			for i in range(n_folders):
				sub_out_folder = os.path.join(out_folder, sub+'_{}'.format(i+1))
				if not os.path.exists(sub_out_folder):
					os.makedirs(sub_out_folder)

				for t in range(n_samples_for_each_video):
					in_img_name = os.path.join(sub_in_folder, image_list[i*n_samples_for_each_video + t])
					out_img_name = os.path.join(sub_out_folder, 'image_{:0>5d}.jpg'.format(t+1))
					print(in_img_name, out_img_name)
					copyfile(in_img_name, out_img_name)

			n_samples_for_last_video = len(image_list) - n_folders*n_samples_for_each_video
			if n_samples_for_last_video > min_n_samples:
				sub_out_folder = os.path.join(out_folder, sub+'_{}'.format(n_folders+1))
				if not os.path.exists(sub_out_folder):
					os.makedirs(sub_out_folder)

				for t in range(n_samples_for_last_video):
					in_img_name = os.path.join(sub_in_folder, image_list[n_folders*n_samples_for_each_video + t])
					out_img_name = os.path.join(sub_out_folder, 'image_{:0>5d}.jpg'.format(t+1))
					print(in_img_name, out_img_name)
					copyfile(in_img_name, out_img_name)


### get clip of whole humane body and create TrainTestlist for 3D-Net
def main_windows_dataset1():
	base_folder = '/media/qcxu/qcxuDisk/windows_datasets_all/'
	__action__ = ['clean', 'normal', 'pick', 'scratch']

	lt = annotation_tools()
	
	for act in __action__:
		for i in range(0, 1000):
			video_id = act + '{:0>6d}'.format(i)
			anno_name = base_folder + 'annotations/' + video_id + '_IndividualStates.txt'
			video_name = base_folder + 'videos/' + act + '/' + video_id + '.avi'
			clip_folder = base_folder + 'clips/' + act + '/' + video_id
			if not os.path.isfile(anno_name) or not os.path.isfile(video_name):
				continue

			lt.vis_anno_result(video_name, anno_name)
			lt.get_clip_from_annos(video_name, anno_name, clip_folder)

	# clip_folder = base_folder + 'clips/'
	# Trainlist_name = base_folder + 'TrainTestlist/trainlist01.txt'
	# Testlist_name = base_folder + 'TrainTestlist/testlist01.txt'
	# lt.get_TrainTestlist(clip_folder, Trainlist_name, Testlist_name)

### get clip of whole humane body and be prepared for AlphaPose
def main_windows_dataset2():
	base_folder = '/media/qcxu/qcxuDisk/windows_datasets_all/clips/'
	__action__ = ['clean', 'normal', 'pick', 'scratch']

	lt = annotation_tools()
	
	for act in __action__:
		sub_folder = os.path.join(base_folder, act)
		in_folder = os.path.join(sub_folder, 'clips_non_fixed_nframe')
		out_folder = os.path.join(sub_folder, 'clips')
		lt.split_into_n_samples(in_folder, out_folder)


def main_scratch_dataset():
	base_folder = '/media/qcxu/qcxuDisk/scratch_dataset/'

	lt = annotation_tools()

	# act = 'pick' #'pick' 'scratch'
	# for i in range(11, 48):
	# 	video_id = '{}'.format(i)
	# 	video_name = base_folder + 'videos/Video ' + video_id + '.mp4'
	# 	anno_name = base_folder + act + '/annotations/Video ' + video_id + '_IndividualStates.txt'
	# 	clip_folder = base_folder + act + '/clips/Video_' + video_id
	# 	if not os.path.isfile(anno_name) or not os.path.isfile(video_name):
	# 		continue

	# 	# lt.vis_anno_result(video_name, anno_name)
	# 	lt.get_clip_from_annos(video_name, anno_name, clip_folder)
		
	act = 'others'
	for i in range(11, 48):
		video_id = '{}'.format(i)
		video_name = base_folder + 'videos/Video ' + video_id + '.mp4'
		anno_name1 = base_folder + 'scratch' + '/annotations/Video ' + video_id + '_IndividualStates.txt'
		anno_name2 = base_folder + 'pick' + '/annotations/Video ' + video_id + '_IndividualStates.txt'
		clip_folder = base_folder + act + '/clips/Video_' + video_id
		if not os.path.isfile(video_name):
			continue

		lt.get_others_class_from_annos(video_name, anno_name1, anno_name2, clip_folder)


if __name__ == "__main__":
	main_windows_dataset2()
	# main_scratch_dataset()
