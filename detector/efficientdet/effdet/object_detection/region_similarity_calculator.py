
import jittor as jt
from jittor import init
from jittor import nn
'Region Similarity Calculators for BoxLists.\n\nRegion Similarity Calculators compare a pairwise measure of similarity\nbetween the boxes in two BoxLists.\n'
from abc import ABCMeta
from abc import abstractmethod

def area(boxlist):
    'Computes area of boxes.\n\n    Args:\n        boxlist: BoxList holding N boxes\n\n    Returns:\n        a tensor with shape [N] representing box areas.\n    '
    (y_min, x_min, y_max, x_max) = boxlist.boxes.chunk(4, dim=1)
    out = ((y_max - y_min).squeeze(1) * (x_max - x_min).squeeze(1))
    return out

def intersection(boxlist1, boxlist2):
    'Compute pairwise intersection areas between boxes.\n\n    Args:\n        boxlist1: BoxList holding N boxes\n        boxlist2: BoxList holding M boxes\n\n    Returns:\n        a tensor with shape [N, M] representing pairwise intersections\n    '
    (y_min1, x_min1, y_max1, x_max1) = boxlist1.boxes.chunk(4, dim=1)
    (y_min2, x_min2, y_max2, x_max2) = boxlist2.boxes.chunk(4, dim=1)
    all_pairs_min_ymax = jt.min(y_max1, dim=y_max2.T)
    all_pairs_max_ymin = jt.max(y_min1, dim=y_min2.T)
    intersect_heights = jt.clamp((all_pairs_min_ymax - all_pairs_max_ymin), min_v=0)
    all_pairs_min_xmax = jt.min(x_max1, dim=x_max2.T)
    all_pairs_max_xmin = jt.max(x_min1, dim=x_min2.T)
    intersect_widths = jt.clamp((all_pairs_min_xmax - all_pairs_max_xmin), min_v=0)
    return (intersect_heights * intersect_widths)

def iou(boxlist1, boxlist2):
    'Computes pairwise intersection-over-union between box collections.\n\n    Args:\n        boxlist1: BoxList holding N boxes\n        boxlist2: BoxList holding M boxes\n\n    Returns:\n        a tensor with shape [N, M] representing pairwise iou scores.\n    '
    intersections = intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = ((areas1.unsqueeze(1) + areas2.unsqueeze(0)) - intersections)
    return jt.where((intersections == 0.0), x=jt.zeros_like(intersections), y=(intersections / unions))

class RegionSimilarityCalculator(object):
    'Abstract base class for region similarity calculator.'
    __metaclass__ = ABCMeta

    def compare(self, boxlist1, boxlist2):
        'Computes matrix of pairwise similarity between BoxLists.\n\n        This op (to be overridden) computes a measure of pairwise similarity between\n        the boxes in the given BoxLists. Higher values indicate more similarity.\n\n        Note that this method simply measures similarity and does not explicitly\n        perform a matching.\n\n        Args:\n            boxlist1: BoxList holding N boxes.\n            boxlist2: BoxList holding M boxes.\n\n        Returns:\n            a (float32) tensor of shape [N, M] with pairwise similarity score.\n        '
        return self._compare(boxlist1, boxlist2)

    @abstractmethod
    def _compare(self, boxlist1, boxlist2):
        pass

class IouSimilarity(RegionSimilarityCalculator):
    'Class to compute similarity based on Intersection over Union (IOU) metric.\n\n    This class computes pairwise similarity between two BoxLists based on IOU.\n    '

    def _compare(self, boxlist1, boxlist2):
        'Compute pairwise IOU similarity between the two BoxLists.\n\n        Args:\n          boxlist1: BoxList holding N boxes.\n          boxlist2: BoxList holding M boxes.\n\n        Returns:\n          A tensor with shape [N, M] representing pairwise iou scores.\n        '
        return iou(boxlist1, boxlist2)
