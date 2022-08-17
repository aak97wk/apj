
import jittor as jt
from jittor import init
from jittor import nn

class SiLU(nn.Module):
    'export-friendly version of nn.SiLU()'

    @staticmethod
    def execute(x):
        return (x * jt.sigmoid(x))


def get_activation(name='silu', inplace=True):
    if (name == 'silu'):
        module = SiLU()
    elif (name == 'relu'):
        module = nn.ReLU()
    elif (name == 'lrelu'):
        module = nn.LeakyReLU(scale=0.1)
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))
    return module

# def get_activation(name='silu', inplace=True):
#     if (name == 'silu'):
#         module = nn.SiLU(inplace=inplace)
#     elif (name == 'relu'):
#         module = nn.ReLU()
#     elif (name == 'lrelu'):
#         module = nn.LeakyReLU(scale=0.1)
#     else:
#         raise AttributeError('Unsupported act type: {}'.format(name))
#     return module

class BaseConv(nn.Module):
    'A Conv2d -> Batchnorm -> silu/leaky relu block'

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act='silu'):
        super().__init__()
        pad = ((ksize - 1) // 2)
        self.conv = nn.Conv(in_channels, out_channels, ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm(out_channels)
        self.act = get_activation(act, inplace=True)

    def execute(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class DWConv(nn.Module):
    'Depthwise Conv + Conv'

    def __init__(self, in_channels, out_channels, ksize, stride=1, act='silu'):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def execute(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act='silu'):
        super().__init__()
        hidden_channels = int((out_channels * expansion))
        Conv = (DWConv if depthwise else BaseConv)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = (shortcut and (in_channels == out_channels))

    def execute(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = (y + x)
        return y

class ResLayer(nn.Module):
    'Residual layer with `in_channels` inputs.'

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = (in_channels // 2)
        self.layer1 = BaseConv(in_channels, mid_channels, ksize=1, stride=1, act='lrelu')
        self.layer2 = BaseConv(mid_channels, in_channels, ksize=3, stride=1, act='lrelu')

    def execute(self, x):
        out = self.layer2(self.layer1(x))
        return (x + out)

class SPPBottleneck(nn.Module):
    'Spatial pyramid pooling layer used in YOLOv3-SPP'

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation='silu'):
        super().__init__()
        hidden_channels = (in_channels // 2)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([nn.Pool(ks, stride=1, padding=(ks // 2), op='maximum') for ks in kernel_sizes])
        conv2_channels = (hidden_channels * (len(kernel_sizes) + 1))
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def execute(self, x):
        x = self.conv1(x)
        x = jt.contrib.concat(([x] + [m(x) for m in self.m]), dim=1)
        x = self.conv2(x)
        return x

class CSPLayer(nn.Module):
    'C3 in yolov5, CSP Bottleneck with 3 convolutions'

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act='silu'):
        '\n        Args:\n            in_channels (int): input channels.\n            out_channels (int): output channels.\n            n (int): number of Bottlenecks. Default value: 1.\n        '
        super().__init__()
        hidden_channels = int((out_channels * expansion))
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv((2 * hidden_channels), out_channels, 1, stride=1, act=act)
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        self.m = nn.Sequential(*module_list)

    def execute(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = jt.contrib.concat((x_1, x_2), dim=1)
        return self.conv3(x)

class Focus(nn.Module):
    'Focus width and height information into channel space.'

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act='silu'):
        super().__init__()
        self.conv = BaseConv((in_channels * 4), out_channels, ksize, stride, act=act)

    def execute(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = jt.contrib.concat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return self.conv(x)
