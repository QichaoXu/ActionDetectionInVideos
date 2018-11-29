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

import _init_paths
from opt import opt
from dataloader import Image_loader, VideoDetectionLoader, DataWriter, crop_from_dets, Mscoco, DetectionLoader
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
    def __init__(self):
        self.skeleton_size = 17

        # Load yolo detection model
        print('Loading YOLO model..')
        self.det_model = Darknet("AlphaPose/yolo/cfg/yolov3.cfg")
        self.det_model.load_weights('AlphaPose/models/yolo/yolov3.weights')
        self.det_model.cuda()
        self.det_model.eval()

        # Load pose model
        print('Loading pose model..')
        pose_dataset = Mscoco()
        if args.fast_inference:
            self.pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            self.pose_model = InferenNet(4 * 1 + 1, pose_dataset)
        self.pose_model.cuda()
        self.pose_model.eval()


    def get_skeleton(self, inputpath, outputpath):

        # update inputpath in opt
        print(inputpath)
        args.inputpath = inputpath

        # Load input images
        im_names = [img for img in sorted(os.listdir(inputpath)) if img.endswith('jpg')]
        dataset = Image_loader(im_names, format='yolo')

        # Load detection loader
        test_loader = DetectionLoader(dataset, self.det_model).start()

        final_result = []
        for i in range(dataset.__len__()):
            with torch.no_grad():
                (inp, orig_img, im_name, boxes, scores) = test_loader.read()
                if boxes is None or boxes.nelement() == 0:
                    skeleton_result = None
                else:
                    # Pose Estimation
                    inps, pt1, pt2 = crop_from_dets(inp, boxes)
                    inps = Variable(inps.cuda())

                    hm = self.pose_model(inps)
                    hm_data = hm.cpu().data

                    preds_hm, preds_img, preds_scores = getPrediction(
                            hm_data, pt1, pt2, args.inputResH, args.inputResW, args.outputResH, args.outputResW)

                    skeleton_result = pose_nms(boxes, scores, preds_img, preds_scores)
                
                results = {
                        'imgname': im_name.split('/')[-1],
                        'result': skeleton_result
                    }

                final_result.append(results)

        print('===========================> Finish Model Running.')

        if not os.path.exists(outputpath):
            os.mkdir(outputpath)

        out_file = open(os.path.join(outputpath, 'skeleton.txt'), 'w')
        for im_res in final_result:
            im_name = im_res['imgname']

            out_file.write(im_name)
            if im_res['result'] is not None:
                for human in im_res['result']:
                    kp_preds = human['keypoints']
                    kp_scores = human['kp_score']

                    for n in range(kp_scores.shape[0]):
                        out_file.write(' ' + str(int(kp_preds[n, 0])))
                        out_file.write(' ' + str(int(kp_preds[n, 1])))
                        out_file.write(' ' + str(round(float(kp_scores[n]), 2)))
            out_file.write('\n')
        out_file.close()


if __name__ == "__main__":

    base_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/'
    __action__ = ['others', 'pick', 'scratch']

    # base_folder = '/media/qcxu/qcxuDisk/windows_datasets_all/clips/'
    # __action__ = ['normal', 'clean', 'pick', 'scratch']

    # get skeleton
    Al_skeleton = Alphapose_skeleton()
    for act in __action__:

        base_in_clip_folder = base_folder + act + '/clips/'
        base_skeleton_folder = base_folder + act + '/skeletons/'
        base_out_clip_folder = base_folder + 'hand/' + act + '/'

        for sub_id, sub in enumerate(os.listdir(base_in_clip_folder)):

            # if sub != 'Video_12_4_1':
            #     continue

            in_clip_folder = base_in_clip_folder + sub
            skeleton_folder = base_skeleton_folder + sub

            Al_skeleton.get_skeleton(in_clip_folder, skeleton_folder)
