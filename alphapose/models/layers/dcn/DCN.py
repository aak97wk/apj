
import jittor as jt
from jittor import init
from jittor import nn
from . import DeformConv, ModulatedDeformConv

class DCN(nn.Module):
    '\n    Initialize: inplanes, planes, upscale_factor\n    OUTPUT: (planes // upscale_factor^2) * ht * wd\n    '

    def __init__(self, inplanes, planes, dcn, kernel_size, stride=1, padding=0, bias=False):
        super(DCN, self).__init__()
        fallback_on_stride = dcn.get('FALLBACK_ON_STRIDE', False)
        self.with_modulated_dcn = dcn.get('MODULATED', False)
        if fallback_on_stride:
            self.conv = nn.Conv(inplanes, planes, kernel_size, stride=stride, padding=padding, bias=bias)
        else:
            self.deformable_groups = dcn.get('DEFORM_GROUP', 1)
            if (not self.with_modulated_dcn):
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv_offset = nn.Conv(inplanes, (self.deformable_groups * offset_channels), kernel_size, stride=stride, padding=padding)
            self.conv = conv_op(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, deformable_groups=self.deformable_groups, bias=bias)

    def execute(self, x):
        if self.with_modulated_dcn:
            offset_mask = self.conv_offset(x)
            offset = offset_mask[:, :(18 * self.deformable_groups), :, :]
            mask = offset_mask[:, ((- 9) * self.deformable_groups):, :, :]
            mask = mask.sigmoid()
            out = self.conv(x, offset, mask)
        else:
            offset = self.conv_offset(x)
            out = self.conv(x, offset)
        return out
