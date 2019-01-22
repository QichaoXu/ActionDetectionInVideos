import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import copy

import os, sys
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, 'AlphaPose')
add_path(lib_path)

from opt import opt
# from dataloader import Image_loader, VideoDetectionLoader, DataWriter, crop_from_dets, Mscoco, DetectionLoader
from dataloader import *
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *
from SPPE.src.utils.eval import getPrediction_batch
from SPPE.src.utils.img import load_image
import os
import time
from fn import getTime
import cv2
import random

from pPose_nms import pose_nms, write_json

import json
from yolo.darknet import Darknet

args = opt
args.dataset = 'coco'


class Alphapose_skeleton:
    def __init__(self, cuda_id=0, fast_yolo=False):

        self.time_det = 0.0
        self.time_run = 0.0

        self.cuda_id = cuda_id
        self.target_kps = [5, 6, 7, 8, 9, 10]

        # Load yolo detection model
        print('Loading YOLO model..')
        if fast_yolo:
            self.det_model = Darknet('./AlphaPose/yolo/cfg/yolov3-tiny.cfg', self.cuda_id)
            self.det_model.load_weights('./AlphaPose/models/yolo/yolov3-tiny.weights')
        else:
            self.det_model = Darknet('./AlphaPose/yolo/cfg/yolov3.cfg', self.cuda_id)
            self.det_model.load_weights('./AlphaPose/models/yolo/yolov3.weights')
            
        self.det_model.cuda(self.cuda_id)
        self.det_model.eval()

        # Load pose model
        print('Loading Alphapose pose model..')
        pose_dataset = Mscoco()
        if args.fast_inference:
            self.pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            self.pose_model = InferenNet(4 * 1 + 1, pose_dataset)
        self.pose_model.cuda(self.cuda_id)
        self.pose_model.eval()


    def run(self, folder_or_imglist, sample_rate):
        time_run_start = time.time()

        if type(folder_or_imglist) == 'str':
            inputpath = folder_or_imglist
            print(inputpath)
            args.inputpath = inputpath

            # Load input images
            im_names = [img for img in sorted(os.listdir(inputpath)) if img.endswith('jpg')]
            N = len(im_names)
            dataset = Image_loader(im_names, format='yolo')
        else:
            N = len(folder_or_imglist)
            imglist = [img for i, img in enumerate(folder_or_imglist) if i % sample_rate == 0]
            dataset = Image_loader_from_images(imglist, format='yolo')

        # Load detection loader
        test_loader = DetectionLoader(dataset, self.det_model, self.cuda_id).start()
        skeleton_result_list = []
        for i in range(dataset.__len__()):
            with torch.no_grad():
                (inp, orig_img, im_name, boxes, scores) = test_loader.read()

                if boxes is None or boxes.nelement() == 0:
                    skeleton_result = None
                else:
                    # Pose Estimation
                    time_det_start = time.time()
                    inps, pt1, pt2 = crop_from_dets(inp, boxes)
                    inps = Variable(inps.cuda(self.cuda_id))

                    hm = self.pose_model(inps)
                    hm_data = hm.cpu().data

                    preds_hm, preds_img, preds_scores = getPrediction(
                            hm_data, pt1, pt2, args.inputResH, args.inputResW, args.outputResH, args.outputResW)

                    skeleton_result = pose_nms(boxes, scores, preds_img, preds_scores)
                    self.time_det += (time.time() - time_det_start)

                skeleton_result_list.append(skeleton_result)

        skeleton_list = []
        j = 0
        for i in range(N):
            im_name = 'image_{:05d}.jpg'.format(i+1)

            if (i == sample_rate * (1+j)):
                j += 1
            skeleton_result = skeleton_result_list[j]

            skeleton_list.append([im_name.split('/')[-1]])
            if skeleton_result is not None:
                for human in skeleton_result:
                    kp_preds = human['keypoints']
                    kp_scores = human['kp_score']

                    # ## remove small hand 
                    # if float(kp_scores[9]) < 0.2 and float(kp_scores[10]) < 0.2:
                    #     continue

                    for n in range(kp_scores.shape[0]):
                        skeleton_list[-1].append(int(kp_preds[n, 0]))
                        skeleton_list[-1].append(int(kp_preds[n, 1]))
                        skeleton_list[-1].append(round(float(kp_scores[n]), 2))

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
    skeleton_det = Alphapose_skeleton()

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