
import jittor as jt
from jittor import init
import numpy as np
from jittor import nn
from models import net_utils
from models.backbone.sqeezenet import DilationLayer, FeatExtractorSqueezeNetx16
from models.psroi_pooling.modules.psroi_pool import PSRoIPool

class Model(nn.Module):
    feat_stride = 4

    def __init__(self, extractor='squeezenet', pretrained=False, transform_input=False):
        super(Model, self).__init__()
        if (extractor == 'squeezenet'):
            feature_extractor = FeatExtractorSqueezeNetx16(pretrained)
        else:
            assert False, 'invalid feature extractor: {}'.format(extractor)
        self.feature_extractor = feature_extractor
        in_channels = self.feature_extractor.n_feats[(- 1)]
        self.stage_0 = nn.Sequential(nn.Dropout(), nn.Conv(in_channels, 256, 3, padding=1), nn.ReLU())
        n_feats = self.feature_extractor.n_feats[1:]
        in_channels = 256
        out_cs = [128, 256]
        for i in range(1, len(n_feats)):
            out_channels = out_cs[(- i)]
            setattr(self, 'upconv_{}'.format(i), nn.Sequential(nn.Conv(in_channels, out_channels, 3, padding=1, bias=True), nn.BatchNorm(out_channels), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear')))
            feat_channels = n_feats[((- 1) - i)]
            setattr(self, 'proj_{}'.format(i), nn.Sequential(net_utils.ConcatAddTable(DilationLayer(feat_channels, (out_channels // 2), 3, dilation=1), DilationLayer(feat_channels, (out_channels // 2), 5, dilation=1)), nn.Conv((out_channels // 2), (out_channels // 2), 1), nn.BatchNorm((out_channels // 2)), nn.ReLU()))
            in_channels = (out_channels + (out_channels // 2))
        roi_size = 7
        self.cls_conv = nn.Sequential(nn.Conv(in_channels, in_channels, 3, padding=1), nn.BatchNorm(in_channels), nn.ReLU(), nn.Conv(in_channels, (roi_size * roi_size), 1, padding=1))
        self.psroipool_cls = PSRoIPool(roi_size, roi_size, (1.0 / self.feat_stride), roi_size, 1)
        self.avg_pool = nn.Pool(roi_size, stride=roi_size, op='mean')

    def get_cls_score(self, cls_feat, rois):
        '\n\n        :param cls_feat: [N, rsize*rsize, H, W]\n        :param rois: [N, 5] (batch_id, x1, y1, x2, y2)\n        :return: [N], [N]\n        '
        cls_scores = self.psroipool_cls(cls_feat, rois)
        cls_scores = self.avg_pool(cls_scores).view((- 1))
        cls_probs = jt.sigmoid(cls_scores)
        return (cls_scores, cls_probs)

    def get_cls_score_numpy(self, cls_feat, rois):
        '\n\n        :param cls_feat: [1, rsize*rsize, H, W]\n        :param rois: numpy array [N, 4] ( x1, y1, x2, y2)\n        :return: [N], [N]\n        '
        n_rois = rois.shape[0]
        if (n_rois <= 0):
            return np.empty([0])
        _rois = np.zeros([n_rois, 5], dtype=np.float32)
        _rois[:, 1:5] = rois.astype(np.float32)
        _rois = Variable(jt.array(_rois)).cuda(cls_feat.get_device())
        cls_scores = self.psroipool_cls(cls_feat, _rois)
        cls_scores = self.avg_pool(cls_scores).view((- 1))
        cls_probs = jt.sigmoid(cls_scores).numpy()
        return cls_probs

    def execute(self, x, gts=None):
        feats = self.feature_extractor(x)
        x_in = self.stage_0(feats[(- 1)])
        n_feats = self.feature_extractor.n_feats[1:]
        for i in range(1, len(n_feats)):
            x_depth_out = getattr(self, 'upconv_{}'.format(i))(x_in)
            x_project = getattr(self, 'proj_{}'.format(i))(feats[((- 1) - i)])
            x_in = jt.contrib.concat((x_depth_out, x_project), dim=1)
        x_cls_in = x_in
        cls_feat = self.cls_conv(x_cls_in)
        return cls_feat
