import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

import _init_paths
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
import test

if __name__ == '__main__':

    opt = parse_opts()
    
    opt.video_path = os.path.join(opt.base_folder, 'C3D_clips')
    opt.annotation_path = os.path.join(opt.base_folder, 'ucf101_01.json')
    opt.result_path = opt.base_folder

    opt.root_path = './3D-ResNet/data'
    opt.resume_path = os.path.join(opt.root_path, 'results-scratch-18/save_200.pth')
    opt.pretrain_path = os.path.join(opt.root_path, 'models/resnet-18-kinetics.pth')

    opt.dataset = 'ucf101'
    opt.n_classes = 400
    opt.n_finetune_classes = 3
    opt.ft_begin_index = 4
    opt.model = 'resnet'
    opt.model_depth = 18
    opt.resnet_shortcut = 'A'
    opt.sample_duration = 45
    opt.batch_size = 1
    opt.n_threads = 1
    opt.checkpoint = 5

    # opt.scales = [opt.initial_scale]
    # for i in range(1, opt.n_scales):
    #     opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)

    model, parameters = generate_model(opt)
    print(model)


    # criterion = nn.CrossEntropyLoss()
    # if not opt.no_cuda:
    #     criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    # if not opt.no_train:
    #     assert opt.train_crop in ['random', 'corner', 'center']
    #     if opt.train_crop == 'random':
    #         crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    #     elif opt.train_crop == 'corner':
    #         crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    #     elif opt.train_crop == 'center':
    #         crop_method = MultiScaleCornerCrop(
    #             opt.scales, opt.sample_size, crop_positions=['c'])
    #     spatial_transform = Compose([
    #         crop_method,
    #         RandomHorizontalFlip(),
    #         ToTensor(opt.norm_value), norm_method
    #     ])
    #     temporal_transform = TemporalRandomCrop(opt.sample_duration)
    #     target_transform = ClassLabel()
    #     training_data = get_training_set(opt, spatial_transform,
    #                                      temporal_transform, target_transform)
    #     train_loader = torch.utils.data.DataLoader(
    #         training_data,
    #         batch_size=opt.batch_size,
    #         shuffle=True,
    #         num_workers=opt.n_threads,
    #         pin_memory=True)
    #     train_logger = Logger(
    #         os.path.join(opt.result_path, 'train.log'),
    #         ['epoch', 'loss', 'acc', 'lr'])
    #     train_batch_logger = Logger(
    #         os.path.join(opt.result_path, 'train_batch.log'),
    #         ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

    #     if opt.nesterov:
    #         dampening = 0
    #     else:
    #         dampening = opt.dampening
    #     optimizer = optim.SGD(
    #         parameters,
    #         lr=opt.learning_rate,
    #         momentum=opt.momentum,
    #         dampening=dampening,
    #         weight_decay=opt.weight_decay,
    #         nesterov=opt.nesterov)
    #     scheduler = lr_scheduler.ReduceLROnPlateau(
    #         optimizer, 'min', patience=opt.lr_patience)
    # if not opt.no_val:
    #     spatial_transform = Compose([
    #         Scale(opt.sample_size),
    #         CenterCrop(opt.sample_size),
    #         ToTensor(opt.norm_value), norm_method
    #     ])
    #     temporal_transform = LoopPadding(opt.sample_duration)
    #     target_transform = ClassLabel()
    #     validation_data = get_validation_set(
    #         opt, spatial_transform, temporal_transform, target_transform)
    #     val_loader = torch.utils.data.DataLoader(
    #         validation_data,
    #         batch_size=opt.batch_size,
    #         shuffle=False,
    #         num_workers=opt.n_threads,
    #         pin_memory=True)
    #     val_logger = Logger(
    #         os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # if not opt.no_train:
        #     optimizer.load_state_dict(checkpoint['optimizer'])

    # print('run')
    # for i in range(opt.begin_epoch, opt.n_epochs + 1):
    #     if not opt.no_train:
    #         train_epoch(i, train_loader, model, criterion, optimizer, opt,
    #                     train_logger, train_batch_logger)
    #     if not opt.no_val:
    #         validation_loss = val_epoch(i, val_loader, model, criterion, opt,
    #                                     val_logger)

    #     if not opt.no_train and not opt.no_val:
    #         scheduler.step(validation_loss)

    opt.test = True
    if opt.test:
        spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            ToTensor(opt.norm_value), norm_method
        ])
        # spatial_transform = Compose([
        #     HardScale(int(opt.sample_size)),
        #     ToTensor(opt.norm_value), norm_method
        # ])
        temporal_transform = LoopPadding(opt.sample_duration)

        # target_transform = VideoID()
        target_transform = ClassLabel()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test.test(test_loader, model, opt, test_data.class_names)
