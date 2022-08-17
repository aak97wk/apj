
import jittor as jt
from jittor import init
from jittor import nn
"Faster RCNN box coder.\n\nFaster RCNN box coder follows the coding schema described below:\n  ty = (y - ya) / ha\n  tx = (x - xa) / wa\n  th = log(h / ha)\n  tw = log(w / wa)\n  where x, y, w, h denote the box's center coordinates, width and height\n  respectively. Similarly, xa, ya, wa, ha denote the anchor's center\n  coordinates, width and height. tx, ty, tw and th denote the anchor-encoded\n  center, width and height respectively.\n\n  See http://arxiv.org/abs/1506.01497 for details.\n"
from . import box_coder
from . import box_list
EPS = 1e-08

class FasterRcnnBoxCoder(box_coder.BoxCoder):
    'Faster RCNN box coder.'

    def __init__(self, scale_factors=None):
        'Constructor for FasterRcnnBoxCoder.\n\n        Args:\n            scale_factors: List of 4 positive scalars to scale ty, tx, th and tw.\n                If set to None, does not perform scaling. For Faster RCNN,\n                the open-source implementation recommends using [10.0, 10.0, 5.0, 5.0].\n        '
        if scale_factors:
            assert (len(scale_factors) == 4)
            for scalar in scale_factors:
                assert (scalar > 0)
        self._scale_factors = scale_factors

    @property
    def code_size(self):
        return 4

    def _encode(self, boxes, anchors):
        'Encode a box collection with respect to anchor collection.\n\n        Args:\n            boxes: BoxList holding N boxes to be encoded.\n            anchors: BoxList of anchors.\n\n        Returns:\n            a tensor representing N anchor-encoded boxes of the format [ty, tx, th, tw].\n        '
        (ycenter_a, xcenter_a, ha, wa) = anchors.get_center_coordinates_and_sizes()
        (ycenter, xcenter, h, w) = boxes.get_center_coordinates_and_sizes()
        ha += EPS
        wa += EPS
        h += EPS
        w += EPS
        tx = ((xcenter - xcenter_a) / wa)
        ty = ((ycenter - ycenter_a) / ha)
        tw = jt.log((w / wa))
        th = jt.log((h / ha))
        if self._scale_factors:
            ty *= self._scale_factors[0]
            tx *= self._scale_factors[1]
            th *= self._scale_factors[2]
            tw *= self._scale_factors[3]
        return jt.stack([ty, tx, th, tw]).T

    def _decode(self, rel_codes, anchors):
        'Decode relative codes to boxes.\n\n        Args:\n            rel_codes: a tensor representing N anchor-encoded boxes.\n            anchors: BoxList of anchors.\n\n        Returns:\n            boxes: BoxList holding N bounding boxes.\n        '
        (ycenter_a, xcenter_a, ha, wa) = anchors.get_center_coordinates_and_sizes()
        (ty, tx, th, tw) = rel_codes.T.unbind()
        if self._scale_factors:
            ty /= self._scale_factors[0]
            tx /= self._scale_factors[1]
            th /= self._scale_factors[2]
            tw /= self._scale_factors[3]
        w = (jt.exp(tw) * wa)
        h = (jt.exp(th) * ha)
        ycenter = ((ty * ha) + ycenter_a)
        xcenter = ((tx * wa) + xcenter_a)
        ymin = (ycenter - (h / 2.0))
        xmin = (xcenter - (w / 2.0))
        ymax = (ycenter + (h / 2.0))
        xmax = (xcenter + (w / 2.0))
        return box_list.BoxList(jt.stack([ymin, xmin, ymax, xmax]).T)
