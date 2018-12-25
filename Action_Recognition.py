import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

import os, sys
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, 'C3D_ResNet')
add_path(lib_path)

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, HardScale, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from train import train_epoch
from validation import val_epoch

from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import cv2

class Action_Recognition:
    def __init__(self, model_file, sample_duration, cuda_id=0):

        self.opt = parse_opts()

        self.opt.root_path = './C3D_ResNet/data'
        self.opt.resume_path = os.path.join(self.opt.root_path, model_file)
        self.opt.pretrain_path = os.path.join(self.opt.root_path, 'models/resnet-18-kinetics.pth')

        self.opt.cuda_id = cuda_id
        self.opt.dataset = 'ucf101'
        self.opt.n_classes = 400
        self.opt.n_finetune_classes = 3
        self.opt.ft_begin_index = 4
        self.opt.model = 'resnet'
        self.opt.model_depth = 18
        self.opt.resnet_shortcut = 'A'
        self.opt.sample_duration = sample_duration
        self.opt.batch_size = 1
        self.opt.n_threads = 1
        self.opt.checkpoint = 5

        self.opt.arch = '{}-{}'.format(self.opt.model, self.opt.model_depth)
        self.opt.mean = get_mean(self.opt.norm_value, dataset=self.opt.mean_dataset)
        self.opt.std = get_std(self.opt.norm_value)
        # print(self.opt)

        self.model, parameters = generate_model(self.opt)
        # print(self.model)

        if self.opt.no_mean_norm and not self.opt.std_norm:
            norm_method = Normalize([0, 0, 0], [1, 1, 1])
        elif not self.opt.std_norm:
            norm_method = Normalize(self.opt.mean, [1, 1, 1])
        else:
            norm_method = Normalize(self.opt.mean, self.opt.std)

        if self.opt.resume_path:
            print('loading checkpoint {}'.format(self.opt.resume_path))
            checkpoint = torch.load(self.opt.resume_path)
            assert self.opt.arch == checkpoint['arch']

            self.opt.begin_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
       
        self.spatial_transform = Compose([
            Scale(int(self.opt.sample_size / self.opt.scale_in_test)),
            CornerCrop(self.opt.sample_size, self.opt.crop_position_in_test),
            ToTensor(self.opt.norm_value), norm_method
        ])

        self.target_transform = ClassLabel()

        self.model.eval()

    def run(self, clip):
        '''
        input: clips is continuous frames with length T
        return: action recognition probability
        '''
        # prepare dataset
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        clip = clip.unsqueeze(0)
        if self.opt.cuda_id is None:
            inputs = Variable(clip)
        else:
            inputs = Variable(clip.cuda(self.opt.cuda_id))

        # run model
        outputs = self.model(inputs)
        outputs = F.softmax(outputs)

        sorted_scores, locs = torch.topk(outputs, k=3)
        return int(locs[0][0]), outputs[0].detach().cpu().numpy().tolist()



if __name__ == '__main__':

    model_file = 'results-scratch-18-static_BG/save_200.pth'
    reg = Action_Recognition(model_file, cuda_id=1)

    for t in range(100):
        clip = []
        base_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/hand_static_BG/pick/pick_Video_12_4_1_1'
        for i in range(1, 46):
            image_name = 'image_{:05d}.jpg'.format(i)

            path = os.path.join(base_folder, image_name)
            img = cv2.imread(path)
            image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  

            clip.append(image)

        label, probs = reg.run(clip)
        print(label, probs)