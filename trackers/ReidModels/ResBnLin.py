
import jittor as jt
from jittor import init
from jittor import nn
'\n@author:  tanghy\n@contact: thutanghy@gmail.com\n'
from ReidModels.ResNet import build_resnet_backbone
from ReidModels.bn_linear import BNneckLinear

class SpatialAttn(nn.Module):
    'Spatial Attention Layer'

    def __init__(self):
        super(SpatialAttn, self).__init__()

    def execute(self, x):
        x = x.mean(1, keepdims=True)
        h = x.shape[2]
        w = x.shape[3]
        x = x.view((x.shape[0], (- 1)))
        z = x
        for b in range(x.shape[0]):
            z[b] /= jt.sum(z[b])
        z = z.view((x.shape[0], 1, h, w))
        return z

class ResModel(nn.Module):

    def __init__(self, n_ID):
        super().__init__()
        self.backbone = build_resnet_backbone()
        self.head = BNneckLinear(n_ID)
        self.atten = SpatialAttn()
        self.conv1 = nn.Conv(17, 17, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.pool = nn.Pool(2, stride=2, padding=0, op='mean')

    def execute(self, input, posemap, map_weight):
        '\n        See :class:`ReIDHeads.forward`.\n        '
        feat = self.backbone(input)
        (b, c, h, w) = feat.shape
        att = self.conv1(torch * (posemap, map_weight))
        att = nn.relu(att)
        att = self.pool(att)
        att = self.conv1(att)
        att = nn.softmax(att)
        att = self.atten(att)
        att = att.expand(b, c, h, w)
        _feat = torch * (feat, att)
        feat = (_feat + feat)
        return self.head(feat)
