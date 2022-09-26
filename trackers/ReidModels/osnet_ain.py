from __future__ import division, absolute_import
import jittor as jt
from jittor import init
from jittor import nn
import warnings
__all__ = ['osnet_ain_x1_0']
pretrained_urls = {'osnet_ain_x1_0': 'https://drive.google.com/uc?id=1-CaioD9NaqbHK_kzSMW8VE4_3KcsRjEo'}

class ConvLayer(nn.Module):
    'Convolution layer (conv + bn + relu).'

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, IN=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, groups=groups)
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()

    def execute(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return nn.relu(x)

class Conv1x1(nn.Module):
    '1x1 convolution + bn + relu.'

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv(in_channels, out_channels, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()

    def execute(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return nn.relu(x)

class Conv1x1Linear(nn.Module):
    '1x1 convolution + bn (w/o non-linearity).'

    def __init__(self, in_channels, out_channels, stride=1, bn=True):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm(out_channels)

    def execute(self, x):
        x = self.conv(x)
        if (self.bn is not None):
            x = self.bn(x)
        return x

class Conv3x3(nn.Module):
    '3x3 convolution + bn + relu.'

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()

    def execute(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return nn.relu(x)

class LightConv3x3(nn.Module):
    'Lightweight 3x3 convolution.\n    1x1 (linear) + dw 3x3 (nonlinear).\n    '

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()

    def execute(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return nn.relu(x)

class LightConvStream(nn.Module):
    'Lightweight convolution stream.'

    def __init__(self, in_channels, out_channels, depth):
        super(LightConvStream, self).__init__()
        assert (depth >= 1), 'depth must be equal to or larger than 1, but got {}'.format(depth)
        layers = []
        layers += [LightConv3x3(in_channels, out_channels)]
        for i in range((depth - 1)):
            layers += [LightConv3x3(out_channels, out_channels)]
        self.layers = nn.Sequential(*layers)

    def execute(self, x):
        return self.layers(x)

class ChannelGate(nn.Module):
    'A mini-network that generates channel-wise gates conditioned on input tensor.'

    def __init__(self, in_channels, num_gates=None, return_gates=False, gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate, self).__init__()
        if (num_gates is None):
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv(in_channels, (in_channels // reduction), 1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm(((in_channels // reduction), 1, 1))
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv((in_channels // reduction), num_gates, 1, bias=True, padding=0)
        if (gate_activation == 'sigmoid'):
            self.gate_activation = nn.Sigmoid()
        elif (gate_activation == 'relu'):
            self.gate_activation = nn.ReLU()
        elif (gate_activation == 'linear'):
            self.gate_activation = None
        else:
            raise RuntimeError('Unknown gate activation: {}'.format(gate_activation))

    def execute(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if (self.norm1 is not None):
            x = self.norm1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        if (self.gate_activation is not None):
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return (input * x)

class OSBlock(nn.Module):
    'Omni-scale feature learning block.'

    def __init__(self, in_channels, out_channels, reduction=4, T=4, **kwargs):
        super(OSBlock, self).__init__()
        assert (T >= 1)
        assert ((out_channels >= reduction) and ((out_channels % reduction) == 0))
        mid_channels = (out_channels // reduction)
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, (T + 1)):
            self.conv2.append(LightConvStream(mid_channels, mid_channels, t))
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if (in_channels != out_channels):
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def execute(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = (x2 + self.gate(x2_t))
        x3 = self.conv3(x2)
        if (self.downsample is not None):
            identity = self.downsample(identity)
        out = (x3 + identity)
        return nn.relu(out)

class OSBlockINin(nn.Module):
    'Omni-scale feature learning block with instance normalization.'

    def __init__(self, in_channels, out_channels, reduction=4, T=4, **kwargs):
        super(OSBlockINin, self).__init__()
        assert (T >= 1)
        assert ((out_channels >= reduction) and ((out_channels % reduction) == 0))
        mid_channels = (out_channels // reduction)
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, (T + 1)):
            self.conv2.append(LightConvStream(mid_channels, mid_channels, t))
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels, bn=False)
        self.downsample = None
        if (in_channels != out_channels):
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def execute(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = (x2 + self.gate(x2_t))
        x3 = self.conv3(x2)
        x3 = self.IN(x3)
        if (self.downsample is not None):
            identity = self.downsample(identity)
        out = (x3 + identity)
        return nn.relu(out)

class OSNet(nn.Module):
    'Omni-Scale Network.\n    \n    Reference:\n        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.\n        - Zhou et al. Learning Generalisable Omni-Scale Representations\n          for Person Re-Identification. arXiv preprint, 2019.\n    '

    def __init__(self, num_classes, blocks, layers, channels, feature_dim=512, loss='softmax', conv1_IN=False, **kwargs):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert (num_blocks == len(layers))
        assert (num_blocks == (len(channels) - 1))
        self.loss = loss
        self.feature_dim = feature_dim
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=conv1_IN)
        self.maxpool = nn.Pool(3, stride=2, padding=1, op='maximum')
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1])
        self.pool2 = nn.Sequential(Conv1x1(channels[1], channels[1]), nn.Pool(2, stride=2, op='mean'))
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2])
        self.pool3 = nn.Sequential(Conv1x1(channels[2], channels[2]), nn.Pool(2, stride=2, op='mean'))
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3])
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(self.feature_dim, channels[3], dropout_p=None)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _make_layer(self, blocks, layer, in_channels, out_channels):
        layers = []
        layers += [blocks[0](in_channels, out_channels)]
        for i in range(1, len(blocks)):
            layers += [blocks[i](out_channels, out_channels)]
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if ((fc_dims is None) or (fc_dims < 0)):
            self.feature_dim = input_dim
            return None
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim, affine=None))
            layers.append(nn.ReLU())
            if (dropout_p is not None):
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[(- 1)]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if (m.bias is not None):
                    init.constant_(m.bias, value=0.0)
            elif isinstance(m, nn.BatchNorm2d):
                if isinstance(m.weight, float):
                    continue
                init.constant_(m.weight, value=1)
                init.constant_(m.bias, value=0.0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, value=1.0)
                init.constant_(m.bias, value=0.0)
            elif isinstance(m, nn.InstanceNorm2d):
                init.constant_(m.weight, value=1.0)
                init.constant_(m.bias, value=0.0)
            elif isinstance(m, nn.Linear):
                init.gauss_(m.weight, mean=0.0, std=0.01)
                if (m.bias is not None):
                    init.constant_(m.bias, value=0.0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def execute(self, x, return_featuremaps=False):
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = v.view((v.shape[0], (- 1)))
        if (self.fc is not None):
            v = self.fc(v)
        if (not self.is_train):
            return v
        y = self.classifier(v)
        if (self.loss == 'softmax'):
            return y
        elif (self.loss == 'triplet'):
            return (y, v)
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))

def init_pretrained_weights(model, key=''):
    "Initializes model with pretrained weights.\n    \n    Layers that don't match with pretrained layers in name or size are kept unchanged.\n    "
    import os
    import errno
    import gdown
    from collections import OrderedDict

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(os.getenv(ENV_TORCH_HOME, os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch')))
        return torch_home
    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if (e.errno == errno.EEXIST):
            pass
        else:
            raise
    filename = (key + '_imagenet.pth')
    cached_file = os.path.join(model_dir, filename)
    if (not os.path.exists(cached_file)):
        gdown.download(pretrained_urls[key], cached_file, quiet=False)
    state_dict = jt.load(cached_file)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    (matched_layers, discarded_layers) = ([], [])
    for (k, v) in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if ((k in model_dict) and (model_dict[k].shape == v.shape)):
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_parameters(model_dict)
    if (len(matched_layers) == 0):
        warnings.warn('The pretrained weights from "{}" cannot be loaded, please check the key names manually (** ignored and continue **)'.format(cached_file))
    else:
        print('Successfully loaded imagenet pretrained weights from "{}"'.format(cached_file))
        if (len(discarded_layers) > 0):
            print('** The following layers are discarded due to unmatched keys or layer size: {}'.format(discarded_layers))

def osnet_ain_x1_0(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[[OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin], [OSBlockINin, OSBlock]], layers=[2, 2, 2], channels=[64, 256, 384, 512], loss=loss, conv1_IN=True, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='osnet_ain_x1_0')
    return model
