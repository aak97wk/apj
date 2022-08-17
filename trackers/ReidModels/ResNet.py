
import jittor as jt
from jittor import init
from jittor import nn
'\n@author:  tanghy\n@contact: thutanghy@gmail.com\n'
import logging
import math
model_urls = {18: 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 34: 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 50: 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 101: 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', 152: 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}
__all__ = ['ResNet', 'Bottleneck']

class IBN(nn.Module):

    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int((planes / 2))
        self.half = half1
        half2 = (planes - half1)
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm(half2)

    def execute(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0])
        out2 = self.BN(split[1])
        out = jt.contrib.concat((out1, out2), dim=1)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, with_ibn=False, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv(inplanes, planes, 1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        self.conv3 = nn.Conv(planes, (planes * 4), 1, bias=False)
        self.bn3 = nn.BatchNorm((planes * 4))
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if (self.downsample is not None):
            residual = self.downsample(x)
        out += residual
        out = nn.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, last_stride, with_ibn, with_se, block, layers):
        scale = 64
        self.inplanes = scale
        super().__init__()
        self.conv1 = nn.Conv(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.Pool(3, stride=2, padding=1, op='maximum')
        self.layer1 = self._make_layer(block, scale, layers[0], with_ibn=with_ibn)
        self.layer2 = self._make_layer(block, (scale * 2), layers[1], stride=2, with_ibn=with_ibn)
        self.layer3 = self._make_layer(block, (scale * 4), layers[2], stride=2, with_ibn=with_ibn)
        self.layer4 = self._make_layer(block, (scale * 8), layers[3], stride=last_stride)
        self.random_init()

    def _make_layer(self, block, planes, blocks, stride=1, with_ibn=False):
        downsample = None
        if ((stride != 1) or (self.inplanes != (planes * block.expansion))):
            downsample = nn.Sequential(nn.Conv(self.inplanes, (planes * block.expansion), 1, stride=stride, bias=False), nn.BatchNorm((planes * block.expansion)))
        layers = []
        if (planes == 512):
            with_ibn = False
        layers.append(block(self.inplanes, planes, with_ibn, stride, downsample))
        self.inplanes = (planes * block.expansion)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, with_ibn))
        return nn.Sequential(*layers)

    def execute(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = ((m.kernel_size[0] * m.kernel_size[1]) * m.out_channels)
                init.gauss_(m.weight, mean=0, std=math.sqrt((2.0 / n)))
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, value=1)
                init.constant_(m.bias, value=0)

def build_resnet_backbone(pretrain_path='', last_stride=1, with_ibn=False, with_se=False, depth=50):
    '\n    Create a ResNet instance from config.\n    Returns:\n        ResNet: a :class:`ResNet` instance.\n    '
    pretrain = True
    pretrain_path = pretrain_path
    last_stride = last_stride
    with_ibn = with_ibn
    with_se = with_se
    depth = depth
    num_blocks_per_stage = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]
    model = ResNet(last_stride, with_ibn, with_se, Bottleneck, num_blocks_per_stage)
    if pretrain:
        if (not with_ibn):
            state_dict = model_zoo.load_url(model_urls[depth])
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
        else:
            state_dict = jt.load(pretrain_path)['state_dict']
            state_dict.pop('module.fc.weight')
            state_dict.pop('module.fc.bias')
            new_state_dict = {}
            for k in state_dict:
                new_k = '.'.join(k.split('.')[1:])
                if (model.state_dict()[new_k].shape == state_dict[k].shape):
                    new_state_dict[new_k] = state_dict[k]
            state_dict = new_state_dict
        res = model.load_parameters(state_dict, strict=False)
        logger = logging.getLogger(__name__)
        logger.info('missing keys is {}'.format(res.missing_keys))
        logger.info('unexpected keys is {}'.format(res.unexpected_keys))
    return model
