
import jittor as jt
from jittor import init
from jittor import nn
from .._ext import psroi_pooling

class PSRoIPoolingFunction(Function):

    def __init__(self, pooled_height, pooled_width, spatial_scale, group_size, output_dim):
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.group_size = int(group_size)
        self.output_dim = int(output_dim)
        self.output = None
        self.mappingchannel = None
        self.rois = None
        self.feature_size = None

    def execute(self, features, rois):
        (batch_size, num_channels, data_height, data_width) = features.shape
        num_rois = rois.shape[0]
        output = features.new().resize_(num_rois, self.output_dim, self.pooled_height, self.pooled_width).zero_()
        mappingchannel = torch.IntTensor(num_rois, self.output_dim, self.pooled_height, self.pooled_width).zero_().cuda(features.get_device())
        rtn = psroi_pooling.psroi_pooling_forward_cuda(self.pooled_height, self.pooled_width, self.spatial_scale, self.group_size, self.output_dim, features, rois, output, mappingchannel)
        assert (rtn > 0)
        self.output = output
        self.mappingchannel = mappingchannel
        self.rois = rois
        self.feature_size = features.shape
        return output

    def backward(self, grad_output):
        assert ((self.feature_size is not None) and grad_output.is_cuda)
        (batch_size, num_channels, data_height, data_width) = self.feature_size
        grad_input = jt.zeros((batch_size, num_channels, data_height, data_width))
        psroi_pooling.psroi_pooling_backward_cuda(self.pooled_height, self.pooled_width, self.spatial_scale, self.output_dim, grad_output, self.rois, grad_input, self.mappingchannel)
        return (grad_input, None)
