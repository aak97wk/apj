
import jittor as jt
from jittor import init
from jittor import nn
from .darknet import Darknet
from .network_blocks import BaseConv

class YOLOFPN(nn.Module):
    '\n    YOLOFPN module. Darknet 53 is the default backbone of this model.\n    '

    def __init__(self, depth=53, in_features=['dark3', 'dark4', 'dark5']):
        super().__init__()
        self.backbone = Darknet(depth)
        self.in_features = in_features
        self.out1_cbl = self._make_cbl(512, 256, 1)
        self.out1 = self._make_embedding([256, 512], (512 + 256))
        self.out2_cbl = self._make_cbl(256, 128, 1)
        self.out2 = self._make_embedding([128, 256], (256 + 128))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _make_cbl(self, _in, _out, ks):
        return BaseConv(_in, _out, ks, stride=1, act='lrelu')

    def _make_embedding(self, filters_list, in_filters):
        m = nn.Sequential(*[self._make_cbl(in_filters, filters_list[0], 1), self._make_cbl(filters_list[0], filters_list[1], 3), self._make_cbl(filters_list[1], filters_list[0], 1), self._make_cbl(filters_list[0], filters_list[1], 3), self._make_cbl(filters_list[1], filters_list[0], 1)])
        return m

    def load_pretrained_model(self, filename='./weights/darknet53.mix.pth'):
        with open(filename, 'rb') as f:
            state_dict = jt.load(f)
        print('loading pretrained weights...')
        self.backbone.load_parameters(state_dict)

    def execute(self, inputs):
        '\n        Args:\n            inputs (Tensor): input image.\n\n        Returns:\n            Tuple[Tensor]: FPN output features..\n        '
        out_features = self.backbone(inputs)
        (x2, x1, x0) = [out_features[f] for f in self.in_features]
        x1_in = self.out1_cbl(x0)
        x1_in = self.upsample(x1_in)
        x1_in = jt.contrib.concat([x1_in, x1], dim=1)
        out_dark4 = self.out1(x1_in)
        x2_in = self.out2_cbl(out_dark4)
        x2_in = self.upsample(x2_in)
        x2_in = jt.contrib.concat([x2_in, x2], dim=1)
        out_dark3 = self.out2(x2_in)
        outputs = (out_dark3, out_dark4, x0)
        return outputs
