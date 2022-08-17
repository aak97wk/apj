
import jittor as jt
from jittor import init
from jittor import nn
from jittor import models
from .builder import SPPE
from .layers.DUC import DUC
from .layers.SE_Resnet import SEResnet

@SPPE.register_module
class FastPose(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(FastPose, self).__init__()
        self._preset_cfg = cfg['PRESET']
        if ('CONV_DIM' in cfg.keys()):
            self.conv_dim = cfg['CONV_DIM']
        else:
            self.conv_dim = 128
        if ('DCN' in cfg.keys()):
            stage_with_dcn = cfg['STAGE_WITH_DCN']
            dcn = cfg['DCN']
            self.preact = SEResnet(f"resnet{cfg['NUM_LAYERS']}", dcn=dcn, stage_with_dcn=stage_with_dcn)
        else:
            self.preact = SEResnet(f"resnet{cfg['NUM_LAYERS']}")
        assert (cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152])
        x = eval(f"models.resnet{cfg['NUM_LAYERS']}(pretrained=True)") # tycoer
        # import torchvision.models as tm
        # x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")
        model_state = self.preact.state_dict()
        state = {k: v for (k, v) in x.state_dict().items() if ((k in self.preact.state_dict()) and (v.shape == self.preact.state_dict()[k].shape))}
        model_state.update(state)
        self.preact.load_parameters(model_state) #TODO 解注
        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2, norm_layer=norm_layer)
        if (self.conv_dim == 256):
            self.duc2 = DUC(256, 1024, upscale_factor=2, norm_layer=norm_layer)
        else:
            self.duc2 = DUC(256, 512, upscale_factor=2, norm_layer=norm_layer)
        self.conv_out = nn.Conv(self.conv_dim, self._preset_cfg['NUM_JOINTS'], 3, stride=1, padding=1)

    def execute(self, x):
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)
        out = self.conv_out(out)
        return out

    def _initialize(self):
        for m in self.conv_out.modules():
            if isinstance(m, nn.Conv2d):
                init.gauss_(m.weight, std=0.001)
                init.constant_(m.bias, value=0)
