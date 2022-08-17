
import jittor as jt
from jittor import init
from jittor import nn
import sys
from ..functions.psroi_pooling import PSRoIPoolingFunction

class PSRoIPool(Module):

    def __init__(self, pooled_height, pooled_width, spatial_scale, group_size, output_dim):
        super(PSRoIPool, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.group_size = int(group_size)
        self.output_dim = int(output_dim)

    def execute(self, features, rois):
        return PSRoIPoolingFunction(self.pooled_height, self.pooled_width, self.spatial_scale, self.group_size, self.output_dim)(features, rois)
