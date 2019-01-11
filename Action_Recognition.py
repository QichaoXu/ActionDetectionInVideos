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
    Compose, Normalize, HardScale, Scale, ScaleQC, CenterCrop, CornerCrop, MultiScaleCornerCrop,
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
    def __init__(self, model_file, sample_duration, model_type, cuda_id=0):

        self.opt = parse_opts()

        self.opt.model = model_type

        self.opt.root_path = './C3D_ResNet/data'

        self.opt.resume_path = os.path.join(self.opt.root_path, model_file)
        self.opt.pretrain_path = os.path.join(self.opt.root_path, 'models/resnet-18-kinetics.pth')

        self.opt.cuda_id = cuda_id
        self.opt.dataset = 'ucf101'
        self.opt.n_classes = 400
        self.opt.n_finetune_classes = 3
        self.opt.ft_begin_index = 4
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

        print('Loading C3D action-recognition model..')

        self.model, parameters = generate_model(self.opt)
        # print(self.model)

        if self.opt.no_mean_norm and not self.opt.std_norm:
            norm_method = Normalize([0, 0, 0], [1, 1, 1])
        elif not self.opt.std_norm:
            norm_method = Normalize(self.opt.mean, [1, 1, 1])
        else:
            norm_method = Normalize(self.opt.mean, self.opt.std)

        if self.opt.resume_path:
            print('    loading checkpoint {}'.format(self.opt.resume_path))
            checkpoint = torch.load(self.opt.resume_path)
            # assert self.opt.arch == checkpoint['arch']

            self.opt.begin_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])

        self.spatial_transform = Compose([
            ScaleQC(int(self.opt.sample_size / self.opt.scale_in_test)),
            CornerCrop(self.opt.sample_size, self.opt.crop_position_in_test),
            ToTensor(self.opt.norm_value), norm_method
        ])

        self.target_transform = ClassLabel()

        self.model.eval()

    def run(self, clip, heatmap=None):
        '''
        input: clips is continuous frames with length T and batch size N
        return: action recognition probability
        '''

        # prepare dataset
        self.spatial_transform.randomize_parameters()
        # clip = [self.spatial_transform(img) for img in clip_batch for clip_batch in clip]
        # clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        # clip = clip.unsqueeze(0)
        clip_all = []
        for clip_batch in clip:
            clip_all.append(torch.stack([self.spatial_transform(img) for img in clip_batch], 0))
        clip = torch.stack(clip_all, 0).permute(0, 2, 1, 3, 4)

        if self.opt.cuda_id is None:
            inputs = Variable(clip)
        else:
            inputs = Variable(clip.cuda(self.opt.cuda_id))

        if self.opt.model == 'resnet_skeleton':
            # heatmap = [self.spatial_transform(img) for img in heatmap for clip_batch in clip]
            # heatmap = torch.stack(heatmap, 0).permute(1, 0, 2, 3)
            # heatmap = heatmap.unsqueeze(0)
            heatmap_all = []
            for heatmap_batch in heatmap:
                heatmap_all.append(torch.stack([self.spatial_transform(img) for img in heatmap_batch], 0))
            heatmap = torch.stack(heatmap_all, 0).permute(0, 2, 1, 3, 4)

            if self.opt.cuda_id is None:
                heatmap = Variable(heatmap)
            else:
                heatmap = Variable(heatmap.cuda(self.opt.cuda_id))

        # run model
        if self.opt.model == 'resnet_skeleton':
            outputs = self.model(inputs, heatmap)
        else:
            outputs = self.model(inputs)
        outputs = F.softmax(outputs, dim=1)

        # sorted_scores, locs = torch.topk(outputs, k=3)
        # return int(locs[0][0]), outputs[0].detach().cpu().numpy().tolist()
        sorted_scores, locs = torch.topk(outputs, k=1, dim=1)
        labels = locs.detach().cpu().numpy().tolist()
        probs = outputs.detach().cpu().numpy().tolist()
        result_labels = []
        for i, label in enumerate(labels):
            result_labels.append([label[0], probs[i]])

        return result_labels

    def runCAM(self, out_image_name, clip, heatmap=None):

        # hook the feature extractor
        finalconv_name = 'layer4'
        features_blobs = []
        def __hook_feature(module, input, output):
            features_blobs.append(output.data.cpu().numpy())

        print(self.model.module._modules.get(finalconv_name))
        self.model.module._modules.get(finalconv_name).register_forward_hook(__hook_feature)

        # get the softmax weight
        print(self.model)
        params = list(self.model.parameters())
        weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

        def returnCAM(feature_conv, weight_softmax, class_idx):
            # generate the class activation maps upsample to 256x256
            size_upsample = (256, 256)
            bz, nc, h, w = feature_conv.shape
            output_cam = []
            for idx in class_idx:
                cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
                cam = cam.reshape(h, w)
                cam = cam - np.min(cam)
                cam_img = cam / np.max(cam)
                cam_img = np.uint8(255 * cam_img)
                output_cam.append(cv2.resize(cam_img, size_upsample))
            return output_cam

        # download the imagenet category list
        classes = {0:'clean', 1:'normal', 2:'scratch'}
        ori_cam_img_list = []
        for t in range(30):
            ori_cam_img_list.append(cv2.cvtColor(np.asarray(clip[0][t]), cv2.COLOR_BGR2RGB))

        # prepare dataset
        self.spatial_transform.randomize_parameters()
        # clip = [self.spatial_transform(img) for img in clip_batch for clip_batch in clip]
        # clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        # clip = clip.unsqueeze(0)
        clip_all = []
        for clip_batch in clip:
            clip_all.append(torch.stack([self.spatial_transform(img) for img in clip_batch], 0))
        clip = torch.stack(clip_all, 0).permute(0, 2, 1, 3, 4)

        if self.opt.cuda_id is None:
            inputs = Variable(clip)
        else:
            inputs = Variable(clip.cuda(self.opt.cuda_id))

        if self.opt.model == 'resnet_skeleton':
            # heatmap = [self.spatial_transform(img) for img in heatmap for clip_batch in clip]
            # heatmap = torch.stack(heatmap, 0).permute(1, 0, 2, 3)
            # heatmap = heatmap.unsqueeze(0)
            heatmap_all = []
            for heatmap_batch in heatmap:
                heatmap_all.append(torch.stack([self.spatial_transform(img) for img in heatmap_batch], 0))
            heatmap = torch.stack(heatmap_all, 0).permute(0, 2, 1, 3, 4)

            if self.opt.cuda_id is None:
                heatmap = Variable(heatmap)
            else:
                heatmap = Variable(heatmap.cuda(self.opt.cuda_id))

        # run model
        if self.opt.model == 'resnet_skeleton':
            outputs = self.model(inputs, heatmap)
        else:
            outputs = self.model(inputs)

        h_x = F.softmax(outputs, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()

        # output the prediction
        for i in range(0, 3):
            print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

        # generate class activation mapping for the top1 prediction
        print(features_blobs[0].shape)
        features_blobs[0] = np.mean(features_blobs[0], axis=2)
        print(features_blobs[0].shape)
        print(weight_softmax.shape)
        print(idx[0])
        CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

        # render the CAM and output
        img = clip[0][0]
        height, width, _ = ori_cam_img_list[0].shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)

        cv2.namedWindow('result')
        flag = True
        while flag:
            for t in range(30):
                result = heatmap * 0.3 + ori_cam_img_list[t] * 0.5
                result = result.astype('uint8')

                cv2.imshow('result', result)
                c = cv2.waitKey(0) % 256
                if c == 13:
                    flag = False
                    break

        cv2.destroyAllWindows()
        if is_heatmap:
            cv2.imwrite('CAM-skeleton.jpg', result)
            print('output CAM-skeleton.jpg for the top1 prediction: %s'%classes[idx[0]])
        else:
            cv2.imwrite('CAM.jpg', result)
            print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
        return



if __name__ == '__main__':
    
    is_heatmap = False

    T = 30
    if is_heatmap:
        reg_model_file = 'results-scratch-18-static_BG-30-skeleton/save_200.pth'
        model_type = 'resnet_skeleton'
    else:
        reg_model_file = 'results-scratch-18-static_BG-30/save_200.pth'
        model_type = 'resnet'

    reg = Action_Recognition(reg_model_file, sample_duration=T, model_type=model_type, cuda_id=0)

    clip = []
    heatmap = []
    base_folder = '/media/qcxu/qcxuDisk/Dataset/scratch_dataset/hand_static_BG/scratch/scratch_Video_11_1_1_1'
    if is_heatmap:
        out_image_name = 'CAM-' + base_folder.split('/')[-1] + '.jpg'
    else:
        out_image_name = 'CAM-' + base_folder.split('/')[-1] + '-skeleton.jpg'
    print(out_image_name)
    for i in range(1, T+1):
        image_name = 'image_{:05d}.jpg'.format(i)
        path = os.path.join(base_folder, image_name)
        img = cv2.imread(path)
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  
        clip.append(image)

        image_name = 'image_{:05d}_heatmap.jpg'.format(i)
        path = os.path.join(base_folder, image_name)
        img = cv2.imread(path)
        image = Image.fromarray(img[:, :, 0])  
        heatmap.append(image)

    if is_heatmap:
        result_labels = reg.runCAM(out_image_name, [clip], [heatmap])
    else:
        result_labels = reg.runCAM(out_image_name, [clip])
    print(result_labels)