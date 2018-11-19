from skeleton_tools import skeleton_tools
import os
import sys
import subprocess
import json


st = skeleton_tools()
sample_duration = 45



base_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/videos/'
video_name = 'Video_11.mp4'
video_length = 130 # seconds

### video path
video_file_path = base_folder + video_name

### out path during process
out_folder = base_folder + video_name.split('.')[0]
if not os.path.exists(out_folder):
	os.makedirs(out_folder)

### extracted hand-clip folder
base_out_clip_folder = out_folder + '/' + 'C3D_clips/'
if not os.path.exists(base_out_clip_folder):
	os.makedirs(base_out_clip_folder)

### image folder
image_folder = out_folder + '/' + 'images'
if not os.path.exists(image_folder):
	os.makedirs(image_folder)

def get_clip():

	## convet video to image
	cmd = 'ffmpeg -i \"{}\" \"{}/image_%05d.jpg\"'.format(
		video_file_path, image_folder)
	print(cmd)
	print('\n')
	subprocess.call(cmd, shell=True)

	num_frames = len(os.listdir(image_folder))
	frame_id = 1
	for clip_id in range(num_frames//sample_duration):
		t_string = '{:03d}'.format(clip_id)

		### temporal folder
		tmp_folder = out_folder + '/' + t_string
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
		out_clip_folder = base_out_clip_folder + '/others/'
		if not os.path.exists(out_clip_folder):
			os.makedirs(out_clip_folder)

		## move image to temporal folder
		for t in range(1, sample_duration+1):

			in_img = image_folder + '/image_{:05d}.jpg'.format(frame_id)
			out_img = in_clip_folder + '/image_{:05d}.jpg'.format(t)
			frame_id += 1

			cmd = 'cp \"{}\"  \"{}\"'.format(in_img, out_img)
			subprocess.call(cmd, shell=True)

		### calculate skeleton and get hand-clip
		st.get_skeleton(in_clip_folder, skeleton_folder)
		st.get_valid_skeletons(skeleton_folder, is_labeled=False)
		st.get_hand_clip(in_clip_folder, skeleton_folder, out_clip_folder+'/'+t_string, is_labeled=False)

	### get ready for C3D
	st.create_Testlist(base_out_clip_folder, out_folder)


def conver_frame_to_video(image_folder, video_name, frame_rate=30.0):
    # video_name must be end with .avi.

    image_names_list = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".ppm")]
    frame = cv2.imread(os.path.join(image_folder, image_names_list[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width,height))

    for image in image_names_list:
        img = cv2.imread(os.path.join(image_folder, image))
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

    print(video_name, len(image_names_list))

def vis_result():

	### result
	f = open(os.path.join(out_folder, 'val.json'))
	result = json.load(f)

	sum_valid_human = 0
	num_frames = len(os.listdir(image_folder))
	for clip_id in range(num_frames//sample_duration):
		t_string = '{:03d}'.format(clip_id)

		### temporal folder
		tmp_folder = out_folder + '/' + t_string

		### image folder
		in_clip_folder = tmp_folder + '/' + 'images'

		### skeleton folder
		skeleton_folder = tmp_folder + '/' + 'skeletons'

		### extracted hand-clip folder
		out_clip_folder = base_out_clip_folder + '/others/'
		num_valid_human = len([x for x in os.listdir(out_clip_folder) if t_string == x.split('_')[0]])

		valid_keys = [str(x) for x in range(sum_valid_human, sum_valid_human+num_valid_human)]
		result_t = [result[key] for key in valid_keys]
		sum_valid_human += num_valid_human

		print(num_valid_human)

		### visualize result
		st.vis_skeleton(in_clip_folder, skeleton_folder, result_t, is_labeled=False, is_save=True)


if __name__ == "__main__":
	
	# get_clip()
	
	vis_result()