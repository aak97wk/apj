
import jittor as jt
from jittor import init
from jittor import nn
from .builder import SPPE
from .layers.Resnet import ResNet

@SPPE.register_module
class SimplePose(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(SimplePose, self).__init__()
        self._preset_cfg = cfg['PRESET']
        self.deconv_dim = cfg['NUM_DECONV_FILTERS']
        self._norm_layer = norm_layer
        self.preact = ResNet(f"resnet{cfg['NUM_LAYERS']}")
        assert (cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152])
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")
        model_state = self.preact.state_dict()
        state = {k: v for (k, v) in x.state_dict().items() if ((k in self.preact.state_dict()) and (v.shape == self.preact.state_dict()[k].shape))}
        model_state.update(state)
        self.preact.load_parameters(model_state)
        self.deconv_layers = self._make_deconv_layer()
        self.final_layer = nn.Conv(self.deconv_dim[2], self._preset_cfg['NUM_JOINTS'], 1, stride=1, padding=0)

    def _make_deconv_layer(self):
        deconv_layers = []
        deconv1 = nn.ConvTranspose(2048, self.deconv_dim[0], 4, stride=2, padding=(int((4 / 2)) - 1), bias=False)
        bn1 = self._norm_layer(self.deconv_dim[0])
        deconv2 = nn.ConvTranspose(self.deconv_dim[0], self.deconv_dim[1], 4, stride=2, padding=(int((4 / 2)) - 1), bias=False)
        bn2 = self._norm_layer(self.deconv_dim[1])
        deconv3 = nn.ConvTranspose(self.deconv_dim[1], self.deconv_dim[2], 4, stride=2, padding=(int((4 / 2)) - 1), bias=False)
        bn3 = self._norm_layer(self.deconv_dim[2])
        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU())
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU())
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU())
        return nn.Sequential(*deconv_layers)

    def _initialize(self):
        for (name, m) in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                init.gauss_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, value=1)
                init.constant_(m.bias, value=0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                init.gauss_(m.weight, std=0.001)
                init.constant_(m.bias, value=0)

    def execute(self, x):
        out = self.preact(x)
        out = self.deconv_layers(out)
        out = self.final_layer(out)
        return out
