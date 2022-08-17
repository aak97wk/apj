
import jittor as jt
from jittor import init
from jittor import nn
'Base box coder.\n\nBox coders convert between coordinate frames, namely image-centric\n(with (0,0) on the top left of image) and anchor-centric (with (0,0) being\ndefined by a specific anchor).\n\nUsers of a BoxCoder can call two methods:\n encode: which encodes a box with respect to a given anchor\n  (or rather, a tensor of boxes wrt a corresponding tensor of anchors) and\n decode: which inverts this encoding with a decode operation.\nIn both cases, the arguments are assumed to be in 1-1 correspondence already;\nit is not the job of a BoxCoder to perform matching.\n'
from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
FASTER_RCNN = 'faster_rcnn'
KEYPOINT = 'keypoint'
MEAN_STDDEV = 'mean_stddev'
SQUARE = 'square'

class BoxCoder(object):
    'Abstract base class for box coder.'
    __metaclass__ = ABCMeta

    @abstractproperty
    def code_size(self):
        'Return the size of each code.\n\n        This number is a constant and should agree with the output of the `encode`\n        op (e.g. if rel_codes is the output of self.encode(...), then it should have\n        shape [N, code_size()]).  This abstractproperty should be overridden by\n        implementations.\n\n        Returns:\n          an integer constant\n        '
        pass

    def encode(self, boxes, anchors):
        'Encode a box list relative to an anchor collection.\n\n        Args:\n          boxes: BoxList holding N boxes to be encoded\n          anchors: BoxList of N anchors\n\n        Returns:\n          a tensor representing N relative-encoded boxes\n        '
        return self._encode(boxes, anchors)

    def decode(self, rel_codes, anchors):
        'Decode boxes that are encoded relative to an anchor collection.\n\n        Args:\n          rel_codes: a tensor representing N relative-encoded boxes\n          anchors: BoxList of anchors\n\n        Returns:\n          boxlist: BoxList holding N boxes encoded in the ordinary way (i.e.,\n            with corners y_min, x_min, y_max, x_max)\n        '
        return self._decode(rel_codes, anchors)

    @abstractmethod
    def _encode(self, boxes, anchors):
        'Method to be overridden by implementations.\n\n        Args:\n          boxes: BoxList holding N boxes to be encoded\n          anchors: BoxList of N anchors\n\n        Returns:\n          a tensor representing N relative-encoded boxes\n        '
        pass

    @abstractmethod
    def _decode(self, rel_codes, anchors):
        'Method to be overridden by implementations.\n\n        Args:\n          rel_codes: a tensor representing N relative-encoded boxes\n          anchors: BoxList of anchors\n\n        Returns:\n          boxlist: BoxList holding N boxes encoded in the ordinary way (i.e.,\n            with corners y_min, x_min, y_max, x_max)\n        '
        pass

def batch_decode(encoded_boxes, box_coder, anchors):
    'Decode a batch of encoded boxes.\n\n    This op takes a batch of encoded bounding boxes and transforms\n    them to a batch of bounding boxes specified by their corners in\n    the order of [y_min, x_min, y_max, x_max].\n\n    Args:\n        encoded_boxes: a float32 tensor of shape [batch_size, num_anchors,\n            code_size] representing the location of the objects.\n        box_coder: a BoxCoder object.\n        anchors: a BoxList of anchors used to encode `encoded_boxes`.\n\n    Returns:\n        decoded_boxes: a float32 tensor of shape [batch_size, num_anchors, coder_size]\n            representing the corners of the objects in the order of [y_min, x_min, y_max, x_max].\n\n    Raises:\n        ValueError: if batch sizes of the inputs are inconsistent, or if\n        the number of anchors inferred from encoded_boxes and anchors are inconsistent.\n    '
    assert (len(encoded_boxes.shape) == 3)
    if (encoded_boxes.shape[1] != anchors.num_boxes()):
        raise ValueError(('The number of anchors inferred from encoded_boxes and anchors are inconsistent: shape[1] of encoded_boxes %s should be equal to the number of anchors: %s.' % (encoded_boxes.shape[1], anchors.num_boxes())))
    decoded_boxes = jt.stack([box_coder.decode(boxes, anchors).boxes for boxes in encoded_boxes.unbind()])
    return decoded_boxes
