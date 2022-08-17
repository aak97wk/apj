
import jittor as jt
from jittor import init
from jittor import nn
'Base target assigner module.\n\nThe job of a TargetAssigner is, for a given set of anchors (bounding boxes) and\ngroundtruth detections (bounding boxes), to assign classification and regression\ntargets to each anchor as well as weights to each anchor (specifying, e.g.,\nwhich anchors should not contribute to training loss).\n\nIt assigns classification/regression targets by performing the following steps:\n1) Computing pairwise similarity between anchors and groundtruth boxes using a\n  provided RegionSimilarity Calculator\n2) Computing a matching based on the similarity matrix using a provided Matcher\n3) Assigning regression targets based on the matching and a provided BoxCoder\n4) Assigning classification targets based on the matching and groundtruth labels\n\nNote that TargetAssigners only operate on detections from a single\nimage at a time, so any logic for applying a TargetAssigner to multiple\nimages must be handled externally.\n'
from . import box_list
KEYPOINTS_FIELD_NAME = 'keypoints'

class TargetAssigner(object):
    'Target assigner to compute classification and regression targets.'

    def __init__(self, similarity_calc, matcher, box_coder, negative_class_weight=1.0, unmatched_cls_target=None):
        'Construct Object Detection Target Assigner.\n\n        Args:\n            similarity_calc: a RegionSimilarityCalculator\n\n            matcher: Matcher used to match groundtruth to anchors.\n\n            box_coder: BoxCoder used to encode matching groundtruth boxes with respect to anchors.\n\n            negative_class_weight: classification weight to be associated to negative\n                anchors (default: 1.0). The weight must be in [0., 1.].\n\n            unmatched_cls_target: a float32 tensor with shape [d_1, d_2, ..., d_k]\n                which is consistent with the classification target for each\n                anchor (and can be empty for scalar targets).  This shape must thus be\n                compatible with the groundtruth labels that are passed to the "assign"\n                function (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]).\n                If set to None, unmatched_cls_target is set to be [0] for each anchor.\n\n        Raises:\n            ValueError: if similarity_calc is not a RegionSimilarityCalculator or\n                if matcher is not a Matcher or if box_coder is not a BoxCoder\n        '
        self._similarity_calc = similarity_calc
        self._matcher = matcher
        self._box_coder = box_coder
        self._negative_class_weight = negative_class_weight
        self._unmatched_cls_target = unmatched_cls_target

    @property
    def box_coder(self):
        return self._box_coder

    def assign(self, anchors, groundtruth_boxes, groundtruth_labels=None, groundtruth_weights=None, **params):
        'Assign classification and regression targets to each anchor.\n\n        For a given set of anchors and groundtruth detections, match anchors\n        to groundtruth_boxes and assign classification and regression targets to\n        each anchor as well as weights based on the resulting match (specifying,\n        e.g., which anchors should not contribute to training loss).\n\n        Anchors that are not matched to anything are given a classification target\n        of self._unmatched_cls_target which can be specified via the constructor.\n\n        Args:\n            anchors: a BoxList representing N anchors\n\n            groundtruth_boxes: a BoxList representing M groundtruth boxes\n\n            groundtruth_labels:  a tensor of shape [M, d_1, ... d_k]\n                with labels for each of the ground_truth boxes. The subshape\n                [d_1, ... d_k] can be empty (corresponding to scalar inputs).  When set\n                to None, groundtruth_labels assumes a binary problem where all\n                ground_truth boxes get a positive label (of 1).\n\n            groundtruth_weights: a float tensor of shape [M] indicating the weight to\n                assign to all anchors match to a particular groundtruth box. The weights\n                must be in [0., 1.]. If None, all weights are set to 1.\n\n            **params: Additional keyword arguments for specific implementations of the Matcher.\n\n        Returns:\n            cls_targets: a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k],\n                where the subshape [d_1, ..., d_k] is compatible with groundtruth_labels\n                which has shape [num_gt_boxes, d_1, d_2, ... d_k].\n\n            cls_weights: a float32 tensor with shape [num_anchors]\n\n            reg_targets: a float32 tensor with shape [num_anchors, box_code_dimension]\n\n            reg_weights: a float32 tensor with shape [num_anchors]\n\n            match: a matcher.Match object encoding the match between anchors and groundtruth boxes,\n                with rows corresponding to groundtruth boxes and columns corresponding to anchors.\n\n        Raises:\n            ValueError: if anchors or groundtruth_boxes are not of type box_list.BoxList\n        '
        if (not isinstance(anchors, box_list.BoxList)):
            raise ValueError('anchors must be an BoxList')
        if (not isinstance(groundtruth_boxes, box_list.BoxList)):
            raise ValueError('groundtruth_boxes must be an BoxList')
        device = anchors.device
        if (groundtruth_labels is None):
            groundtruth_labels = jt.ones(groundtruth_boxes.num_boxes()).unsqueeze(0)
            groundtruth_labels = groundtruth_labels.unsqueeze((- 1))
        if (groundtruth_weights is None):
            num_gt_boxes = groundtruth_boxes.num_boxes()
            if (not num_gt_boxes):
                num_gt_boxes = groundtruth_boxes.num_boxes()
            groundtruth_weights = jt.ones([num_gt_boxes])
        match_quality_matrix = self._similarity_calc.compare(groundtruth_boxes, anchors)
        match = self._matcher.match(match_quality_matrix, **params)
        reg_targets = self._create_regression_targets(anchors, groundtruth_boxes, match)
        cls_targets = self._create_classification_targets(groundtruth_labels, match)
        reg_weights = self._create_regression_weights(match, groundtruth_weights)
        cls_weights = self._create_classification_weights(match, groundtruth_weights)
        return (cls_targets, cls_weights, reg_targets, reg_weights, match)

    def _create_regression_targets(self, anchors, groundtruth_boxes, match):
        'Returns a regression target for each anchor.\n\n        Args:\n            anchors: a BoxList representing N anchors\n\n            groundtruth_boxes: a BoxList representing M groundtruth_boxes\n\n            match: a matcher.Match object\n\n        Returns:\n            reg_targets: a float32 tensor with shape [N, box_code_dimension]\n        '
        device = anchors.device
        zero_box = jt.zeros(4)
        matched_gt_boxes = match.gather_based_on_match(groundtruth_boxes.boxes, unmatched_value=zero_box, ignored_value=zero_box)
        matched_gt_boxlist = box_list.BoxList(matched_gt_boxes)
        if groundtruth_boxes.has_field(KEYPOINTS_FIELD_NAME):
            groundtruth_keypoints = groundtruth_boxes.get_field(KEYPOINTS_FIELD_NAME)
            zero_kp = jt.zeros(groundtruth_keypoints.shape[1:])
            matched_keypoints = match.gather_based_on_match(groundtruth_keypoints, unmatched_value=zero_kp, ignored_value=zero_kp)
            matched_gt_boxlist.add_field(KEYPOINTS_FIELD_NAME, matched_keypoints)
        matched_reg_targets = self._box_coder.encode(matched_gt_boxlist, anchors)
        unmatched_ignored_reg_targets = self._default_regression_target(device).repeat(match.match_results.shape[0], 1)
        matched_anchors_mask = match.matched_column_indicator()
        reg_targets = jt.where(matched_anchors_mask.unsqueeze(1), x=matched_reg_targets, y=unmatched_ignored_reg_targets)
        return reg_targets

    def _default_regression_target(self, device):
        'Returns the default target for anchors to regress to.\n\n        Default regression targets are set to zero (though in this implementation what\n        these targets are set to should not matter as the regression weight of any box\n        set to regress to the default target is zero).\n\n        Returns:\n            default_target: a float32 tensor with shape [1, box_code_dimension]\n        '
        return jt.zeros((1, self._box_coder.code_size))

    def _create_classification_targets(self, groundtruth_labels, match):
        'Create classification targets for each anchor.\n\n        Assign a classification target of for each anchor to the matching\n        groundtruth label that is provided by match.  Anchors that are not matched\n        to anything are given the target self._unmatched_cls_target\n\n        Args:\n            groundtruth_labels:  a tensor of shape [num_gt_boxes, d_1, ... d_k]\n                with labels for each of the ground_truth boxes. The subshape\n                [d_1, ... d_k] can be empty (corresponding to scalar labels).\n            match: a matcher.Match object that provides a matching between anchors\n                and groundtruth boxes.\n\n        Returns:\n            a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k], where the\n            subshape [d_1, ..., d_k] is compatible with groundtruth_labels which has\n            shape [num_gt_boxes, d_1, d_2, ... d_k].\n        '
        if (self._unmatched_cls_target is not None):
            uct = self._unmatched_cls_target
        else:
            uct = jt.float32(0)
        return match.gather_based_on_match(groundtruth_labels, unmatched_value=uct, ignored_value=uct)

    def _create_regression_weights(self, match, groundtruth_weights):
        'Set regression weight for each anchor.\n\n        Only positive anchors are set to contribute to the regression loss, so this\n        method returns a weight of 1 for every positive anchor and 0 for every\n        negative anchor.\n\n        Args:\n            match: a matcher.Match object that provides a matching between anchors and groundtruth boxes.\n            groundtruth_weights: a float tensor of shape [M] indicating the weight to\n                assign to all anchors match to a particular groundtruth box.\n\n        Returns:\n            a float32 tensor with shape [num_anchors] representing regression weights.\n        '
        zs = jt.float32(0)
        return match.gather_based_on_match(groundtruth_weights, ignored_value=zs, unmatched_value=zs)

    def _create_classification_weights(self, match, groundtruth_weights):
        'Create classification weights for each anchor.\n\n        Positive (matched) anchors are associated with a weight of\n        positive_class_weight and negative (unmatched) anchors are associated with\n        a weight of negative_class_weight. When anchors are ignored, weights are set\n        to zero. By default, both positive/negative weights are set to 1.0,\n        but they can be adjusted to handle class imbalance (which is almost always\n        the case in object detection).\n\n        Args:\n            match: a matcher.Match object that provides a matching between anchors and groundtruth boxes.\n            groundtruth_weights: a float tensor of shape [M] indicating the weight to\n                assign to all anchors match to a particular groundtruth box.\n\n        Returns:\n            a float32 tensor with shape [num_anchors] representing classification weights.\n        '
        ignored = jt.float32(0)
        ncw = jt.float32(self._negative_class_weight)
        return match.gather_based_on_match(groundtruth_weights, ignored_value=ignored, unmatched_value=ncw)

    def get_box_coder(self):
        'Get BoxCoder of this TargetAssigner.\n\n        Returns:\n            BoxCoder object.\n        '
        return self._box_coder
