
import jittor as jt
from jittor import init
from jittor import nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    '3x3 convolution with padding'
    return nn.Conv(in_planes, out_planes, 3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if (norm_layer is None):
            norm_layer = nn.BatchNorm2d
        if ((groups != 1) or (base_width != 64)):
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if (dilation > 1):
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if (self.downsample is not None):
            identity = self.downsample(x)
        out += identity
        out = nn.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d, dcn=None):
        super(Bottleneck, self).__init__()
        self.dcn = dcn
        self.with_dcn = (dcn is not None)
        self.conv1 = nn.Conv(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes, momentum=0.1)
        if self.with_dcn:
            fallback_on_stride = dcn.get('FALLBACK_ON_STRIDE', False)
            self.with_modulated_dcn = dcn.get('MODULATED', False)
        if ((not self.with_dcn) or fallback_on_stride):
            self.conv2 = nn.Conv(planes, planes, 3, stride=stride, padding=1, bias=False)
        else:
            from .dcn import DeformConv, ModulatedDeformConv
            self.deformable_groups = dcn.get('DEFORM_GROUP', 1)
            if (not self.with_modulated_dcn):
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv(planes, (self.deformable_groups * offset_channels), 3, stride=stride, padding=1)
            self.conv2 = conv_op(planes, planes, kernel_size=3, stride=stride, padding=1, deformable_groups=self.deformable_groups, bias=False)
        self.bn2 = norm_layer(planes, momentum=0.1)
        self.conv3 = nn.Conv(planes, (planes * 4), 1, bias=False)
        self.bn3 = norm_layer((planes * 4), momentum=0.1)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        residual = x
        out = nn.relu(self.bn1(self.conv1(x)))
        if (not self.with_dcn):
            out = nn.relu(self.bn2(self.conv2(out)))
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :(18 * self.deformable_groups), :, :]
            mask = offset_mask[:, ((- 9) * self.deformable_groups):, :, :]
            mask = mask.sigmoid()
            out = nn.relu(self.bn2(self.conv2(out, offset, mask)))
        else:
            offset = self.conv2_offset(out)
            out = nn.relu(self.bn2(self.conv2(out, offset)))
        out = self.conv3(out)
        out = self.bn3(out)
        if (self.downsample is not None):
            residual = self.downsample(x)
        out += residual
        out = nn.relu(out)
        return out

class ResNet(nn.Module):
    ' ResNet '

    def __init__(self, architecture, norm_layer=nn.BatchNorm2d, dcn=None, stage_with_dcn=(False, False, False, False)):
        super(ResNet, self).__init__()
        self._norm_layer = norm_layer
        self.architecture = architecture
        assert (architecture in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
        layers = {'resnet18': [2, 2, 2, 2], 'resnet34': [3, 4, 6, 3], 'resnet50': [3, 4, 6, 3], 'resnet101': [3, 4, 23, 3], 'resnet152': [3, 8, 36, 3]}
        self.inplanes = 64
        if ((architecture == 'resnet18') or (architecture == 'resnet34')):
            self.block = BasicBlock
        else:
            self.block = Bottleneck
        self.layers = layers[architecture]
        self.conv1 = nn.Conv(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu = nn.ReLU()
        self.maxpool = nn.Pool(3, stride=2, padding=1, op='maximum')
        stage_dcn = [(dcn if with_dcn else None) for with_dcn in stage_with_dcn]
        self.layer1 = self.make_layer(self.block, 64, self.layers[0], dcn=stage_dcn[0])
        self.layer2 = self.make_layer(self.block, 128, self.layers[1], stride=2, dcn=stage_dcn[1])
        self.layer3 = self.make_layer(self.block, 256, self.layers[2], stride=2, dcn=stage_dcn[2])
        self.layer4 = self.make_layer(self.block, 512, self.layers[3], stride=2, dcn=stage_dcn[3])

    def execute(self, x):
        x = self.maxpool(nn.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def stages(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if ((stride != 1) or (self.inplanes != (planes * block.expansion))):
            downsample = nn.Sequential(nn.Conv(self.inplanes, (planes * block.expansion), 1, stride=stride, bias=False), self._norm_layer((planes * block.expansion)))
        layers = []
        if ((self.architecture == 'resnet18') or (self.architecture == 'resnet34')):
            layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=self._norm_layer))
            self.inplanes = (planes * block.expansion)
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, norm_layer=self._norm_layer))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=self._norm_layer, dcn=dcn))
            self.inplanes = (planes * block.expansion)
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, norm_layer=self._norm_layer))
        return nn.Sequential(*layers)
