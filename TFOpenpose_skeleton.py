
import os, sys
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, 'tf-openpose')
add_path(lib_path)

import sys
import time
import cv2
import numpy as np

import argparse
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


class TFOpenpose_skeleton:
    def __init__(self, cuda_id=0):

        self.time_det = 0.0
        self.time_run = 0.0

        parser = argparse.ArgumentParser(description='tf-pose-estimation run')
        # parser.add_argument('--image', type=str, default='./images/p1.jpg')
        parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')
        parser.add_argument('--resize', type=str, default='656x368',
                            help='if provided, resize images before they are processed. default=0x0, Recommends : 656x368 or 432x368 or 1312x736 ')
        parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                            help='if provided, resize heatmaps before they are post-processed. default=1.0')

        args = parser.parse_args()
        self.w, self.h = model_wh(args.resize)
        self.resize_out_ratio = args.resize_out_ratio
        self.model = TfPoseEstimator(get_graph_path(args.model), target_size=(self.w, self.h))

        # skeletopn keypoint order
        # Nose, LEye, REye, LEar, REar
        # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
        # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        self.order = [0, 14, 15, 16, 17,
                    2, 5, 3, 6, 4, 7,
                    8, 11, 9, 12, 10, 13]
        # self.order = [0, 15, 16, 17, 18,
        #             2, 5, 3, 6, 4, 7,
        #             9, 12, 10, 13, 11, 14]

    def __kp2list(self, humans, image_h, image_w):
        human_list = []
        for human in humans:
            for i in range(len(self.order)):

                if self.order[i] not in human.body_parts.keys():
                    if len(human_list) == 0:
                        return [] 
                    human_list.append(human_list[-3])
                    human_list.append(human_list[-2])
                    human_list.append(human_list[-1])
                    continue

                body_part = human.body_parts[self.order[i]]
                x = int(body_part.x * image_w + 0.5)
                y = int(body_part.y * image_h + 0.5)
                score = body_part.score
                human_list.append(x)
                human_list.append(y)
                human_list.append(score)
                # cv2.circle(npimg, (x, y), 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)
        return human_list

    def run(self, folder_or_imglist, sample_rate):
        time_run_start = time.time()

        if type(folder_or_imglist) == 'str':
            inputpath = folder_or_imglist
            print(inputpath)

            imglist = []
            for im_name in os.listdir(inputpath):
                if im_name.endswith('jpg'):
                    imglist.append(cv2.imread(os.path.join(inputpath, im_name)))
        else:
            imglist = folder_or_imglist

        skeleton_list = []
        for i, img in enumerate(imglist):
            im_name = 'image_{:05d}.jpg'.format(i+1)

            time_det_start = time.time()
            if i % sample_rate == 0:
                # keypoints, output_image = self.openpose.forward(img, True)
                humans = self.model.inference(img, resize_to_default=(self.w > 0 and self.h > 0), upsample_size=self.resize_out_ratio)
                keypoints = self.__kp2list(humans, img.shape[0], img.shape[1])
                pre_keypoints = keypoints
            else:
                keypoints = pre_keypoints
            self.time_det += (time.time() - time_det_start)

            skeleton_list.append([im_name.split('/')[-1]])
            skeleton_list[-1] += keypoints

        #print(skeleton_list)
        self.time_run += (time.time() - time_run_start)
        return skeleton_list


    def runtime(self):
        return self.time_det, self.time_run

    def save_skeleton(self, skeleton_list, outputpath):

        if not os.path.exists(outputpath):
            os.mkdir(outputpath)

        out_file = open(os.path.join(outputpath, 'skeleton.txt'), 'w')
        for skeleton in skeleton_list:
            out_file.write(' '.join(str(x) for x in skeleton))
            out_file.write('\n')
        out_file.close()


if __name__ == "__main__":

    base_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/'
    __action__ = ['others', 'pick', 'scratch']

    # base_folder = '/media/qcxu/qcxuDisk/windows_datasets_all/clips/'
    # __action__ = ['normal', 'clean', 'pick', 'scratch']

    # get skeleton
    skeleton_det = TFOpenpose_skeleton()

    time1 = time.time()
    for act in __action__:

        base_in_clip_folder = base_folder + act + '/clips/'
        base_skeleton_folder = base_folder + act + '/skeletons/'

        for sub_id, sub in enumerate(os.listdir(base_in_clip_folder)):

            if act != 'pick' or sub != 'Video_12_4_1':
                continue

            in_clip_folder = base_in_clip_folder + sub
            skeleton_folder = base_skeleton_folder + sub

            imglist = []
            for im_name in os.listdir(in_clip_folder):
                if im_name.endswith('jpg'):
                    imglist.append(cv2.imread(os.path.join(in_clip_folder, im_name)))

            skeleton_list = skeleton_det.run(imglist, sample_rate=1)
            # skeleton_det.save_skeleton(skeleton_list, skeleton_folder)
    print(time.time() - time1)