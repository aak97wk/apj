
import jittor as jt
from jittor import init
from jittor import nn

class SELayer(nn.Module):

    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, (channel // reduction)), nn.ReLU(), nn.Linear((channel // reduction), channel), nn.Sigmoid())

    def execute(self, x):
        (b, c, _, _) = x.shape
        y = self.avg_pool(x).view((b, c))
        y = self.fc(y).view((b, c, 1, 1))
        return (x * y)
