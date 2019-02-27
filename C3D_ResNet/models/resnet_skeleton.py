import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'ResNet_skeleton', 'resnet_skeleton10', 'resnet_skeleton18', 'resnet_skeleton34', 'resnet_skeleton50', 'resnet_skeleton101',
    'resnet_skeleton152', 'resnet_skeleton200'
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_skeleton(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 64
        super(ResNet_skeleton, self).__init__()

        ## basic
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))

        ## RGB layer
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)

        ## skeleton layer
        self.inplanes = 64
        self.conv1_skeleton = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.layer1_skeleton = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2_skeleton = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3_skeleton = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4_skeleton = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)

        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(1024 * block.expansion, num_classes) ## for concatenate

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, RGB, skeleton):

        ## RGB
        x = self.conv1(RGB)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        ## skeleton heatmap
        y = self.conv1_skeleton(skeleton)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)

        y = self.layer1_skeleton(y)
        y = self.layer2_skeleton(y)

        ## multiply RGB by heatmap mask
        y = x * y

        ## weighted skeleton heatmap
        y = self.layer3_skeleton(y)
        y = self.layer4_skeleton(y)

        ## RGB
        x = self.layer3(x)
        x = self.layer4(x)

        # print(x.size())
        # print(y.size())

        ### fused features
        ## method 1). weighted sum up
        # x = 0.8*x + 0.2*y
        # x = self.avgpool(x)

        ## method 2). concatenate
        x = self.avgpool(x)
        y = self.avgpool(y)
        x = torch.cat((x, y), dim=1)

        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())

        out = self.fc(x)

        return out


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                # print(ft_module, k)
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet_skeleton10(**kwargs):
    """Constructs a ResNet_skeleton-18 model.
    """
    model = ResNet_skeleton(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet_skeleton18(**kwargs):
    """Constructs a ResNet_skeleton-18 model.
    """
    model = ResNet_skeleton(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet_skeleton34(**kwargs):
    """Constructs a ResNet_skeleton-34 model.
    """
    model = ResNet_skeleton(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet_skeleton50(**kwargs):
    """Constructs a ResNet_skeleton-50 model.
    """
    model = ResNet_skeleton(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet_skeleton101(**kwargs):
    """Constructs a ResNet_skeleton-101 model.
    """
    model = ResNet_skeleton(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet_skeleton152(**kwargs):
    """Constructs a ResNet_skeleton-101 model.
    """
    model = ResNet_skeleton(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet_skeleton200(**kwargs):
    """Constructs a ResNet_skeleton-101 model.
    """
    model = ResNet_skeleton(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
