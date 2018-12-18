# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import os, sys
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, 'MSRAPose', 'lib')
add_path(lib_path)

from core.config import config
from core.config import update_config
from core.config import update_dir
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models

import cv2
import numpy as np
from utils.transforms import get_affine_transform
from core.inference import get_final_preds

############ for yolo
from dataloader import *
from yolo.darknet import Darknet


class MSRA_skeleton():
    def __init__(self):

        self.time_det = 0.0
        self.num_joints = 17
        self.target_kps = [5, 6, 7, 8, 9, 10]

        # Load yolo detection model
        print('Loading YOLO model..')
        self.det_model = Darknet("AlphaPose/yolo/cfg/yolov3.cfg")
        self.det_model.load_weights('AlphaPose/models/yolo/yolov3.weights')
        self.det_model.cuda()
        self.det_model.eval()

        cfg_file = 'MSRAPose/experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml' 
        model_file = 'MSRAPose/models/pytorch/pose_coco/pose_resnet_50_256x192.pth.tar'

        # update config
        update_config(cfg_file)
        config.TEST.MODEL_FILE = model_file

        # cudnn related setting
        cudnn.benchmark = config.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = config.CUDNN.ENABLED

        # load pre-trained model
        self.model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
            config, is_train=False
        )
        print('Loading MSRA pose model..')
        print('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        self.model.load_state_dict(torch.load(config.TEST.MODEL_FILE))

        gpus = [int(i) for i in config.GPUS.split(',')]
        self.model = torch.nn.DataParallel(self.model, device_ids=gpus).cuda()
        self.model.eval()

        # image transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])

    def _box2cs(self, box, image_width, image_height):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h, image_width, image_height)

    def _xywh2cs(self, x, y, w, h, image_width, image_height):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        
        aspect_ratio = image_width * 1.0 / image_height
        pixel_std = 200

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array(
            [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def detect_skeleton_on_single_human(self, image, box):
        '''
        input: image read by opencv2
        '''

        data_numpy = image.copy()

        # object detection box        
        if box is None:
            box = [0, 0, data_numpy.shape[0], data_numpy.shape[1]]
        c, s = self._box2cs(box, data_numpy.shape[0], data_numpy.shape[1])
        r = 0

        trans = get_affine_transform(c, s, r, config.MODEL.IMAGE_SIZE)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(config.MODEL.IMAGE_SIZE[0]), int(config.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR
        )

        input = self.transform(input).unsqueeze(0)

        with torch.no_grad():
            # compute output heatmap
            output = self.model(input)
            output = output.clone().cpu().numpy()

            # heatmap = output
            # heatmap_hand = heatmap[0][self.target_kps[0]]
            # print(heatmap.shape)
            # for kk in self.target_kps[1:]:
            #     heatmap_hand += heatmap[0][kk]
            # cv2.imshow('skeletons', heatmap_hand)
            # cv2.waitKey()

            # compute coordinate
            preds, maxvals = get_final_preds(
                config, output, np.asarray([c]), np.asarray([s]))

            return preds[0]

    def run(self, folder_or_imglist):
        if type(folder_or_imglist) == 'str':
            inputpath = folder_or_imglist
            print(inputpath)
            args.inputpath = inputpath

            # Load input images
            im_names = [img for img in sorted(os.listdir(inputpath)) if img.endswith('jpg')]
            dataset = Image_loader(im_names, format='yolo')
        else:
            imglist = folder_or_imglist
            dataset = Image_loader_from_images(imglist, format='yolo')

        # Load detection loader
        test_loader = DetectionLoader(dataset, self.det_model).start()

        skeleton_list = []
        # final_result = []
        for i in range(dataset.__len__()):
            with torch.no_grad():
                (inp, orig_img, im_name, boxes, scores) = test_loader.read()
                
                skeleton_result = []
                if boxes is None or boxes.nelement() == 0:
                    skeleton_result = None
                else:
                    # Pose Estimation
                    time1 = time.time()
                    for box in boxes.tolist():
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        box = [x1, y1, x2-x1, y2-y1]
                        skeleton_result.append(self.detect_skeleton_on_single_human(orig_img, box))
                    self.time_det += (time.time() - time1)

                skeleton_list.append([im_name.split('/')[-1]])
                if skeleton_result is not None:
                    for human in skeleton_result:
                        for mat in human:
                            skeleton_list[-1].append(int(mat[0]))
                            skeleton_list[-1].append(int(mat[1]))
                            skeleton_list[-1].append(0.8)

        return skeleton_list

    def runtime(self):
        return self.time_det

    def generate_target_points(self, joints, image_size, sigma):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''        
        # target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        # target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros((self.num_joints,
                           image_size[1],
                           image_size[0]),
                          dtype=np.float32)

        tmp_size = sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = [1, 1]#image_size / image_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= image_size[0] or ul[1] >= image_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                # target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], image_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], image_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], image_size[0])
            img_y = max(0, ul[1]), min(br[1], image_size[1])

            v = 1 #target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target#, target_weight

    def generate_target_lines(self, joints, image_size, target_kps):

        l_pair = [(0, 1), (0, 2), (1, 3), (2, 4),       # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),    # Hand
            (17, 11), (17, 12),                         # Body
            (11, 13), (12, 14), (13, 15), (14, 16)]     # Leg

        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), 
                    (77,255,222), (77,196,255), (77,135,255), (191,255,77), (77,255,77), 
                    (77,222,255), (255,156,127), 
                    (0,127,255), (255,127,77), (0,77,255), (255,77,36)]

        # Nose, LEye, REye, LEar, REar
        # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
        # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                    (77,255,255), (77, 255, 204), (77,204,255), (191, 255, 77), (77,191,255), (191, 255, 77),
                    (204,77,255), (77,255,204), (191,77,255), (77,255,191), (127,77,255), (77,255,127), (0, 255, 255)]

        img = np.zeros(shape=image_size, dtype='uint8')
        part_line = {}
        for n in range(self.num_joints):
            # if float(kp_scores_h[n]) <= 0.05:
            #     continue

            cor_x, cor_y = int(joints[n][0]), int(joints[n][1])
            part_line[n] = (cor_x, cor_y)
            # cv2.circle(img, (cor_x, cor_y), 4, p_color[n], -1)

        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if i not in target_kps:
                continue

            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(img, start_xy, end_xy, line_color[i], 5)
        return img

if __name__ == '__main__':

    # box = [450, 160, 350, 560]

    ms = MSRA_skeleton()

    input_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/pick/clips/Video_12_4_1'
    
    imglist = []
    for img_name in os.listdir(input_folder):
        if img_name.endswith('jpg'):
            imglist.append(cv2.imread(os.path.join(input_folder, img_name)))

    skeleton = ms.run(imglist)

    for img in imglist:

        skeleton_result = ms.detect_skeleton_on_single_human(img, None)
        
        # plot
        for i, mat in enumerate(skeleton_result):
            x, y = int(mat[0]), int(mat[1])
            cv2.circle(img, (x, y), 2, (255, 0, 0), 2)
            # cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

        # vis result
        cv2.imshow('res', img)

        target = ms.generate_target_points(skeleton_result, img.shape, sigma=5)

        target_all = np.sum(target, axis=0)
        cv2.imshow('image3', target_all)

        img_ske = ms.generate_target_lines(skeleton_result, img.shape, target_kps=[5,6,7,8])
        cv2.imshow('imag', img_ske)

        cv2.waitKey(0)

