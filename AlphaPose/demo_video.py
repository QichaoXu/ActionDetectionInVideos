from skeleton_tools import skeleton_tools
import os
import sys
import subprocess
import json
import cv2



class demo_video:
	def __init__(self, st, base_folder, video_name):

		self.st = st

		self.sample_duration = 45

		### video path
		self.video_file_path = base_folder + video_name

		### out path during process
		self.out_folder = base_folder + video_name.split('.')[0]
		if not os.path.exists(self.out_folder):
			os.makedirs(self.out_folder)

		### extracted hand-clip folder
		self.base_out_clip_folder = self.out_folder + '/' + 'C3D_clips/'
		if not os.path.exists(self.base_out_clip_folder):
			os.makedirs(self.base_out_clip_folder)

		### image folder
		self.image_folder = self.out_folder + '/' + 'images'
		if not os.path.exists(self.image_folder):
			os.makedirs(self.image_folder)

	def get_clip(self):

		## convet video to image
		num_frames = len(os.listdir(self.image_folder))
		if num_frames == 0:
			cmd = 'ffmpeg -i \"{}\" \"{}/image_%05d.jpg\"'.format(
				self.video_file_path, self.image_folder)
			print(cmd)
			print('\n')
			subprocess.call(cmd, shell=True)

		num_frames = len(os.listdir(self.image_folder))
		frame_id = 1
		for clip_id in range(1):#num_frames//self.sample_duration):
			t_string = '{:03d}'.format(clip_id)

			### temporal folder
			tmp_folder = self.out_folder + '/' + t_string
			if not os.path.exists(tmp_folder):
				os.makedirs(tmp_folder)

			### image folder
			in_clip_folder = tmp_folder + '/' + 'images'
			if not os.path.exists(in_clip_folder):
				os.makedirs(in_clip_folder)

			### skeleton folder
			skeleton_folder = tmp_folder + '/' + 'skeletons'
			if not os.path.exists(skeleton_folder):
				os.makedirs(skeleton_folder)

			### extracted hand-clip folder
			out_clip_folder = self.base_out_clip_folder + '/others/'
			if not os.path.exists(out_clip_folder):
				os.makedirs(out_clip_folder)

			## move image to temporal folder
			for t in range(1, self.sample_duration+1):

				in_img = self.image_folder + '/image_{:05d}.jpg'.format(frame_id)
				out_img = in_clip_folder + '/image_{:05d}.jpg'.format(t)
				frame_id += 1

				cmd = 'cp \"{}\"  \"{}\"'.format(in_img, out_img)
				subprocess.call(cmd, shell=True)

			### calculate skeleton and get hand-clip
			self.st.get_skeleton(in_clip_folder, skeleton_folder)
			self.st.get_valid_skeletons(skeleton_folder, is_labeled=False)
			self.st.vis_skeleton(in_clip_folder, skeleton_folder, None, is_labeled=False, is_save=True)
			# self.st.get_hand_clip(in_clip_folder, skeleton_folder, out_clip_folder+'/'+t_string, is_labeled=False)

		### get ready for C3D
		self.st.create_Testlist(self.base_out_clip_folder, self.out_folder)


	def __conver_frame_to_video(self, image_folder, video):
	    # video_name must be end with .avi.

	    image_names_list = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".ppm")]
	    frame = cv2.imread(os.path.join(image_folder, image_names_list[0]))
	    height, width, layers = frame.shape

	    for image in image_names_list:
	        img = cv2.imread(os.path.join(image_folder, image))
	        video.write(img)

	    return video

	def vis_result(self):

		### result
		f = open(os.path.join(self.out_folder, 'val.json'))
		result = json.load(f)

		### result video
		out_video_name = self.out_folder + '/res.avi'
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		video = cv2.VideoWriter(out_video_name, fourcc, 30.0, (1280, 720))

		sum_valid_human = 0
		num_frames = len(os.listdir(self.image_folder))
		for clip_id in range(num_frames//self.sample_duration):
			t_string = '{:03d}'.format(clip_id)

			### temporal folder
			tmp_folder = self.out_folder + '/' + t_string

			### image folder
			in_clip_folder = tmp_folder + '/' + 'images'

			### skeleton folder
			skeleton_folder = tmp_folder + '/' + 'skeletons'

			### extracted hand-clip folder
			out_clip_folder = self.base_out_clip_folder + '/others/'
			num_valid_human = len([x for x in os.listdir(out_clip_folder) if t_string == x.split('_')[0]])

			valid_keys = [str(x) for x in range(sum_valid_human, sum_valid_human+num_valid_human)]
			result_t = [result[key] for key in valid_keys]
			sum_valid_human += num_valid_human

			print(num_valid_human)

			### visualize result
			self.st.vis_skeleton(in_clip_folder, skeleton_folder, result_t, is_labeled=False, is_save=True)
			video = self.__conver_frame_to_video(skeleton_folder+'/res', video)

		cv2.destroyAllWindows()
		video.release()


if __name__ == "__main__":

	st = skeleton_tools()
	base_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/videos/'
	video_name = 'Video_45.mp4'

	dv = demo_video(st, base_folder, video_name)

	dv.get_clip()
	# dv.vis_result()