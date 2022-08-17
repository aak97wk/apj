
import jittor as jt
from jittor import init
from jittor import nn

class DilationLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same_padding', dilation=1, bn=False):
        super(DilationLayer, self).__init__()
        if (padding == 'same_padding'):
            padding = int((((kernel_size - 1) / 2) * dilation))
        self.Dconv = nn.Conv(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.Drelu = nn.ReLU()
        self.Dbn = (nn.BatchNorm(out_channels) if bn else None)

    def execute(self, x):
        x = self.Dconv(x)
        if (self.Dbn is not None):
            x = self.Dbn(x)
        x = self.Drelu(x)
        return x

class FeatExtractorSqueezeNetx16(nn.Module):
    n_feats = [64, 128, 256, 512]

    def __init__(self, pretrained=True):
        super(FeatExtractorSqueezeNetx16, self).__init__()
        print('loading layers from squeezenet1_1...')
        sq = models.squeezenet1_1(pretrained=pretrained)
        self.conv1 = nn.Sequential(sq.features[0], sq.features[1])
        self.conv2 = nn.Sequential(nn.Pool(3, stride=2, padding=1, op='maximum'), sq.features[3], sq.features[4])
        self.conv3 = nn.Sequential(nn.Pool(3, stride=2, padding=1, op='maximum'), sq.features[6], sq.features[7])
        self.conv4 = nn.Sequential(nn.Pool(3, stride=2, padding=1, op='maximum'), sq.features[9], sq.features[10], sq.features[11], sq.features[12])
        self.conv1[0].padding = (1, 1)

    def execute(self, x):
        x2 = self.conv1(x)
        x4 = self.conv2(x2)
        x8 = self.conv3(x4)
        x16 = self.conv4(x8)
        return (x2, x4, x8, x16)
