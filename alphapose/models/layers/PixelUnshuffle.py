
import jittor as jt
from jittor import init
from jittor import nn

class PixelUnshuffle(nn.Module):
    '\n    Initialize: inplanes, planes, upscale_factor\n    OUTPUT: (planes // upscale_factor^2) * ht * wd\n    '

    def __init__(self, downscale_factor=2):
        super(PixelUnshuffle, self).__init__()
        self._r = downscale_factor

    def execute(self, x):
        (b, c, h, w) = x.shape
        out_c = (c * (self._r * self._r))
        out_h = (h // self._r)
        out_w = (w // self._r)
        x_view = x.view((b, c, out_h, self._r, out_w, self._r))
        x_prime = x_view.permute(0, 1, 3, 5, 2, 4).view((b, out_c, out_h, out_w))
        return x_prime
