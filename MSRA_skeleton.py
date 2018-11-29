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

import _init_paths
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


class MSRA_skeleton():
    def __init__(self):
        
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

    def detect_skeleton_on_single_human(self, image_file, box):
        '''
        input: image  path, pre-detcted box indicating human position
        '''

        # Load an image
        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        # object detection box
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

            # compute coordinate
            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), np.asarray([c]), np.asarray([s]))

            # plot
            img = data_numpy.copy()
            for i, mat in enumerate(preds[0]):
                x, y = int(mat[0]), int(mat[1])
                print(x, y)
                cv2.circle(img, (x, y), 2, (255, 0, 0), 2)
                cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

            # vis result
            cv2.imshow('res', img)
            cv2.waitKey(0)

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



if __name__ == '__main__':

    ms = MSRA_skeleton()

    image_file = 'image_00001.jpg'
    box = [450, 160, 350, 560]
    ms.detect_skeleton_on_single_human(image_file, box)
