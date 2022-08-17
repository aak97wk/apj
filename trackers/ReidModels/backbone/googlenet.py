
import jittor as jt
from jittor import init
'GoogLeNet with PyTorch.'
from jittor import nn
from .lrn import SpatialCrossMapLRN

class Inception(nn.Module):

    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        self.b1 = nn.Sequential(nn.Conv(in_planes, n1x1, 1), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv(in_planes, n3x3red, 1), nn.ReLU(), nn.Conv(n3x3red, n3x3, 3, padding=1), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv(in_planes, n5x5red, 1), nn.ReLU(), nn.Conv(n5x5red, n5x5, 5, padding=2), nn.ReLU())
        self.b4 = nn.Sequential(nn.Pool(3, stride=1, padding=1, op='maximum'), nn.Conv(in_planes, pool_planes, 1), nn.ReLU())

    def execute(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return jt.contrib.concat([y1, y2, y3, y4], dim=1)

class GoogLeNet(nn.Module):
    output_channels = 832

    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(nn.Conv(3, 64, 7, stride=2, padding=3), nn.ReLU(), nn.Pool(3, stride=2, ceil_mode=True, op='maximum'), SpatialCrossMapLRN(5), nn.Conv(64, 64, 1), nn.ReLU(), nn.Conv(64, 192, 3, padding=1), nn.ReLU(), SpatialCrossMapLRN(5), nn.Pool(3, stride=2, ceil_mode=True, op='maximum'))
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.Pool(3, stride=2, ceil_mode=True, op='maximum')
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

    def execute(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        return out
