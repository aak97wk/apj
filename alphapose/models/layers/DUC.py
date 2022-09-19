
import jittor as jt
from jittor import init
from jittor import nn

class DUC(nn.Module):
    '\n    Initialize: inplanes, planes, upscale_factor\n    OUTPUT: (planes // upscale_factor^2) * ht * wd\n    '

    def __init__(self, inplanes, planes, upscale_factor=2, norm_layer=nn.BatchNorm2d):
        super(DUC, self).__init__()
        self.conv = nn.Conv(inplanes, planes, 3, padding=1, bias=False)
        self.bn = norm_layer(planes, momentum=0.1)
        self.relu = nn.ReLU()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def execute(self, x):
        print('DUC.py', x.shape) # todo
        x = self.conv(x)
        x = self.bn(x)
        x = nn.relu(x)
        x = self.pixel_shuffle(x)
        return x
