
import jittor as jt
from jittor import init
' RetinaNet / EfficientDet Anchor Gen\n\nAdapted for PyTorch from Tensorflow impl at\n    https://github.com/google/automl/blob/6f6694cec1a48cdb33d5d1551a2d5db8ad227798/efficientdet/anchors.py\n\nHacked together by Ross Wightman, original copyright below\n'
'Anchor definition.\n\nThis module is borrowed from TPU RetinaNet implementation:\nhttps://github.com/tensorflow/tpu/blob/master/models/official/retinanet/anchors.py\n'
import collections
import numpy as np
from jittor import nn
from .object_detection import argmax_matcher
from .object_detection import box_list
from .object_detection import faster_rcnn_box_coder
from .object_detection import region_similarity_calculator
from .object_detection import target_assigner
MIN_CLASS_SCORE = (- 5.0)
_DUMMY_DETECTION_SCORE = (- 100000.0)
MAX_DETECTION_POINTS = 5000
MAX_DETECTIONS_PER_IMAGE = 100

def decode_box_outputs(rel_codes, anchors, output_xyxy=False):
    'Transforms relative regression coordinates to absolute positions.\n\n    Network predictions are normalized and relative to a given anchor; this\n    reverses the transformation and outputs absolute coordinates for the input image.\n\n    Args:\n        rel_codes: box regression targets.\n\n        anchors: anchors on all feature levels.\n\n    Returns:\n        outputs: bounding boxes.\n\n    '
    ycenter_a = ((anchors[0] + anchors[2]) / 2)
    xcenter_a = ((anchors[1] + anchors[3]) / 2)
    ha = (anchors[2] - anchors[0])
    wa = (anchors[3] - anchors[1])
    (ty, tx, th, tw) = rel_codes
    w = (jt.exp(tw) * wa)
    h = (jt.exp(th) * ha)
    ycenter = ((ty * ha) + ycenter_a)
    xcenter = ((tx * wa) + xcenter_a)
    ymin = (ycenter - (h / 2.0))
    xmin = (xcenter - (w / 2.0))
    ymax = (ycenter + (h / 2.0))
    xmax = (xcenter + (w / 2.0))
    if output_xyxy:
        out = jt.stack([xmin, ymin, xmax, ymax], dim=1)
    else:
        out = jt.stack([ymin, xmin, ymax, xmax], dim=1)
    return out

def _generate_anchor_configs(min_level, max_level, num_scales, aspect_ratios):
    'Generates mapping from output level to a list of anchor configurations.\n\n    A configuration is a tuple of (num_anchors, scale, aspect_ratio).\n\n    Args:\n        min_level: integer number of minimum level of the output feature pyramid.\n\n        max_level: integer number of maximum level of the output feature pyramid.\n\n        num_scales: integer number representing intermediate scales added on each level.\n            For instances, num_scales=2 adds two additional anchor scales [2^0, 2^0.5] on each level.\n\n        aspect_ratios: list of tuples representing the aspect ratio anchors added on each level.\n            For instances, aspect_ratios = [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.\n\n    Returns:\n        anchor_configs: a dictionary with keys as the levels of anchors and\n            values as a list of anchor configuration.\n    '
    anchor_configs = {}
    for level in range(min_level, (max_level + 1)):
        anchor_configs[level] = []
        for scale_octave in range(num_scales):
            for aspect in aspect_ratios:
                anchor_configs[level].append(((2 ** level), (scale_octave / float(num_scales)), aspect))
    return anchor_configs

def _generate_anchor_boxes(image_size, anchor_scale, anchor_configs):
    'Generates multiscale anchor boxes.\n\n    Args:\n        image_size: integer number of input image size. The input image has the same dimension for\n            width and height. The image_size should be divided by the largest feature stride 2^max_level.\n\n        anchor_scale: float number representing the scale of size of the base\n            anchor to the feature stride 2^level.\n\n        anchor_configs: a dictionary with keys as the levels of anchors and\n            values as a list of anchor configuration.\n\n    Returns:\n        anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all feature levels.\n\n    Raises:\n        ValueError: input size must be the multiple of largest feature stride.\n    '
    boxes_all = []
    for (_, configs) in anchor_configs.items():
        boxes_level = []
        for config in configs:
            (stride, octave_scale, aspect) = config
            if ((image_size % stride) != 0):
                raise ValueError('input size must be divided by the stride.')
            base_anchor_size = ((anchor_scale * stride) * (2 ** octave_scale))
            anchor_size_x_2 = ((base_anchor_size * aspect[0]) / 2.0)
            anchor_size_y_2 = ((base_anchor_size * aspect[1]) / 2.0)
            x = np.arange((stride / 2), image_size, stride)
            y = np.arange((stride / 2), image_size, stride)
            (xv, yv) = np.meshgrid(x, y)
            xv = xv.reshape((- 1))
            yv = yv.reshape((- 1))
            boxes = np.vstack(((yv - anchor_size_y_2), (xv - anchor_size_x_2), (yv + anchor_size_y_2), (xv + anchor_size_x_2)))
            boxes = np.swapaxes(boxes, 0, 1)
            boxes_level.append(np.expand_dims(boxes, axis=1))
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape([(- 1), 4]))
    anchor_boxes = np.vstack(boxes_all)
    return anchor_boxes

def generate_detections(cls_outputs, box_outputs, anchor_boxes, indices, classes, image_scale, nms_thres=0.5, max_dets=100):
    'Generates detections with RetinaNet model outputs and anchors.\n\n    Args:\n        cls_outputs: a torch tensor with shape [N, 1], which has the highest class\n            scores on all feature levels. The N is the number of selected\n            top-K total anchors on all levels.  (k being MAX_DETECTION_POINTS)\n\n        box_outputs: a torch tensor with shape [N, 4], which stacks box regression\n            outputs on all feature levels. The N is the number of selected top-k\n            total anchors on all levels. (k being MAX_DETECTION_POINTS)\n\n        anchor_boxes: a torch tensor with shape [N, 4], which stacks anchors on all\n            feature levels. The N is the number of selected top-k total anchors on all levels.\n\n        indices: a torch tensor with shape [N], which is the indices from top-k selection.\n\n        classes: a torch tensor with shape [N], which represents the class\n            prediction on all selected anchors from top-k selection.\n\n        image_scale: a float tensor representing the scale between original image\n            and input image for the detector. It is used to rescale detections for\n            evaluating with the original groundtruth annotations.\n\n    Returns:\n        detections: detection results in a tensor with shape [MAX_DETECTION_POINTS, 6],\n            each row representing [x, y, width, height, score, class]\n    '
    anchor_boxes = anchor_boxes[indices, :]
    boxes = decode_box_outputs(box_outputs.T.float(), anchor_boxes.T, output_xyxy=True)
    scores = cls_outputs.sigmoid().squeeze(1).float()
    human_idx = (classes == 0)
    boxes = boxes[human_idx]
    scores = scores[human_idx]
    classes = classes[human_idx]
    top_detection_idx = batched_nms(boxes, scores, classes, iou_threshold=nms_thres)
    top_detection_idx = top_detection_idx[:max_dets]
    boxes = boxes[top_detection_idx]
    scores = scores[(top_detection_idx, None)]
    classes = classes[(top_detection_idx, None)]
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]
    boxes *= image_scale
    classes += 1
    detections = jt.contrib.concat([boxes, scores, classes.float()], dim=1)
    if (len(top_detection_idx) < max_dets):
        detections = jt.contrib.concat([detections, jt.zeros(((max_dets - len(top_detection_idx)), 6), dtype=detections.dtype)], dim=0)
    return detections

class Anchors(nn.Module):
    'RetinaNet Anchors class.'

    def __init__(self, min_level, max_level, num_scales, aspect_ratios, anchor_scale, image_size):
        'Constructs multiscale RetinaNet anchors.\n\n        Args:\n            min_level: integer number of minimum level of the output feature pyramid.\n\n            max_level: integer number of maximum level of the output feature pyramid.\n\n            num_scales: integer number representing intermediate scales added\n                on each level. For instances, num_scales=2 adds two additional\n                anchor scales [2^0, 2^0.5] on each level.\n\n            aspect_ratios: list of tuples representing the aspect ratio anchors added\n                on each level. For instances, aspect_ratios =\n                [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.\n\n            anchor_scale: float number representing the scale of size of the base\n                anchor to the feature stride 2^level.\n\n            image_size: integer number of input image size. The input image has the\n                same dimension for width and height. The image_size should be divided by\n                the largest feature stride 2^max_level.\n        '
        super(Anchors, self).__init__()
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.anchor_scale = anchor_scale
        self.image_size = image_size
        self.config = self._generate_configs()
        self.register_buffer('boxes', self._generate_boxes())

    def _generate_configs(self):
        'Generate configurations of anchor boxes.'
        return _generate_anchor_configs(self.min_level, self.max_level, self.num_scales, self.aspect_ratios)

    def _generate_boxes(self):
        'Generates multiscale anchor boxes.'
        boxes = _generate_anchor_boxes(self.image_size, self.anchor_scale, self.config)
        boxes = jt.array(boxes).float()
        return boxes

    def get_anchors_per_location(self):
        return (self.num_scales * len(self.aspect_ratios))

class AnchorLabeler(nn.Module):
    'Labeler for multiscale anchor boxes.\n    '

    def __init__(self, anchors, num_classes, match_threshold=0.5):
        'Constructs anchor labeler to assign labels to anchors.\n\n        Args:\n            anchors: an instance of class Anchors.\n\n            num_classes: integer number representing number of classes in the dataset.\n\n            match_threshold: float number between 0 and 1 representing the threshold\n                to assign positive labels for anchors.\n        '
        super(AnchorLabeler, self).__init__()
        similarity_calc = region_similarity_calculator.IouSimilarity()
        matcher = argmax_matcher.ArgMaxMatcher(match_threshold, unmatched_threshold=match_threshold, negatives_lower_than_unmatched=True, force_match_for_each_row=True)
        box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()
        self.target_assigner = target_assigner.TargetAssigner(similarity_calc, matcher, box_coder)
        self.anchors = anchors
        self.match_threshold = match_threshold
        self.num_classes = num_classes

    def _unpack_labels(self, labels):
        'Unpacks an array of labels into multiscales labels.'
        labels_unpacked = []
        anchors = self.anchors
        count = 0
        for level in range(anchors.min_level, (anchors.max_level + 1)):
            feat_size = int((anchors.image_size / (2 ** level)))
            steps = ((feat_size ** 2) * anchors.get_anchors_per_location())
            indices = jt.arange(count, start=(count + steps))
            count += steps
            labels_unpacked.append(labels[indices]).view([feat_size, feat_size, (- 1)])
        return labels_unpacked

    def label_anchors(self, gt_boxes, gt_labels):
        'Labels anchors with ground truth inputs.\n\n        Args:\n            gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.\n                For each row, it stores [y0, x0, y1, x1] for four corners of a box.\n\n            gt_labels: A integer tensor with shape [N, 1] representing groundtruth classes.\n\n        Returns:\n            cls_targets_dict: ordered dictionary with keys [min_level, min_level+1, ..., max_level].\n                The values are tensor with shape [height_l, width_l, num_anchors]. The height_l and width_l\n                represent the dimension of class logits at l-th level.\n\n            box_targets_dict: ordered dictionary with keys [min_level, min_level+1, ..., max_level].\n                The values are tensor with shape [height_l, width_l, num_anchors * 4]. The height_l and\n                width_l represent the dimension of bounding box regression output at l-th level.\n\n            num_positives: scalar tensor storing number of positives in an image.\n        '
        gt_box_list = box_list.BoxList(gt_boxes)
        anchor_box_list = box_list.BoxList(self.anchors.boxes)
        (cls_targets, _, box_targets, _, matches) = self.target_assigner.assign(anchor_box_list, gt_box_list, gt_labels)
        cls_targets -= 1
        cls_targets = cls_targets.long()
        cls_targets_dict = self._unpack_labels(cls_targets)
        box_targets_dict = self._unpack_labels(box_targets)
        num_positives = (matches.match_results != (- 1)).float().sum()
        return (cls_targets_dict, box_targets_dict, num_positives)
