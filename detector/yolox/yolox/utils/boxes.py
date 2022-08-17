
import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
from detector.nms.nms_wrapper import nms, multiclass_nms


__all__ = ['filter_box', 'postprocess', 'bboxes_iou', 'matrix_iou', 'adjust_box_anns', 'xyxy2xywh', 'xyxy2cxcywh']

def filter_box(output, scale_range):
    '\n    output: (N, 5+class) shape\n    '
    (min_scale, max_scale) = scale_range
    w = (output[:, 2] - output[:, 0])
    h = (output[:, 3] - output[:, 1])
    keep = (((w * h) > (min_scale * min_scale)) & ((w * h) < (max_scale * max_scale)))
    return output[keep]

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, classes=0, class_agnostic=False):
    box_corner = jt.rand(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - (prediction[:, :, 2] / 2))
    box_corner[:, :, 1] = (prediction[:, :, 1] - (prediction[:, :, 3] / 2))
    box_corner[:, :, 2] = (prediction[:, :, 0] + (prediction[:, :, 2] / 2))
    box_corner[:, :, 3] = (prediction[:, :, 1] + (prediction[:, :, 3] / 2))
    prediction[:, :, :4] = box_corner[:, :, :4]
    output = 0
    for (i, image_pred) in enumerate(prediction):
        if (not image_pred.shape[0]):
            continue
        (class_pred, class_conf) = jt.argmax(image_pred[:, 5:(5 + num_classes)], dim=1, keepdims=True)
        conf_mask = ((image_pred[:, 4] * class_conf.squeeze(-1)) >= conf_thre).flatten()
        detections = jt.contrib.concat((image_pred[:, :5], class_conf, class_pred.float()), dim=1)
        detections = detections[conf_mask]
        if (classes is not None):
            detections = detections[(detections[:, 6:7] == jt.array(classes)).any(1)]
        if (not detections.shape[0]):
            continue
        if class_agnostic:
            nms_out_index = nms(detections[:, :4], (detections[:, 4] * detections[:, 5]), nms_thre)
        else:
            nms_out_index = nms(detections[:, :4], (detections[:, 4] * detections[:, 5]), nms_thre)
        detections = detections[nms_out_index]
        batch_idx = jt.full_like(jt.rand(detections.shape[0], 1), i)
        detections = jt.contrib.concat((batch_idx, detections), dim=1)
        if (isinstance(output, int) and (output == 0)):
            output = detections
        else:
            output = jt.contrib.concat((output, detections))
    return output

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if ((bboxes_a.shape[1] != 4) or (bboxes_b.shape[1] != 4)):
        raise IndexError
    if xyxy:
        tl = jt.max(bboxes_a[:, None, :2], dim=bboxes_b[:, :2])
        br = jt.min(bboxes_a[:, None, 2:], dim=bboxes_b[:, 2:])
        area_a = jt.prod((bboxes_a[:, 2:] - bboxes_a[:, :2]), dim=1)
        area_b = jt.prod((bboxes_b[:, 2:] - bboxes_b[:, :2]), dim=1)
    else:
        tl = jt.max((bboxes_a[:, None, :2] - (bboxes_a[:, None, 2:] / 2)), dim=(bboxes_b[:, :2] - (bboxes_b[:, 2:] / 2)))
        br = jt.min((bboxes_a[:, None, :2] + (bboxes_a[:, None, 2:] / 2)), dim=(bboxes_b[:, :2] + (bboxes_b[:, 2:] / 2)))
        area_a = jt.prod(bboxes_a[:, 2:], dim=1)
        area_b = jt.prod(bboxes_b[:, 2:], dim=1)
    en = (tl < br).astype(tl.dtype).prod(dim=2)
    area_i = (jt.prod((br - tl), dim=2) * en)
    return (area_i / ((area_a[:, None] + area_b) - area_i))

def matrix_iou(a, b):
    '\n    return iou of a and b, numpy version for data augenmentation\n    '
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
    area_i = (np.prod((rb - lt), axis=2) * (lt < rb).all(axis=2))
    area_a = np.prod((a[:, 2:] - a[:, :2]), axis=1)
    area_b = np.prod((b[:, 2:] - b[:, :2]), axis=1)
    return (area_i / (((area_a[:, np.newaxis] + area_b) - area_i) + 1e-12))

def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(((bbox[:, 0::2] * scale_ratio) + padw), 0, w_max)
    bbox[:, 1::2] = np.clip(((bbox[:, 1::2] * scale_ratio) + padh), 0, h_max)
    return bbox

def xyxy2xywh(bboxes):
    bboxes[:, 2] = (bboxes[:, 2] - bboxes[:, 0])
    bboxes[:, 3] = (bboxes[:, 3] - bboxes[:, 1])
    return bboxes

def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = (bboxes[:, 2] - bboxes[:, 0])
    bboxes[:, 3] = (bboxes[:, 3] - bboxes[:, 1])
    bboxes[:, 0] = (bboxes[:, 0] + (bboxes[:, 2] * 0.5))
    bboxes[:, 1] = (bboxes[:, 1] + (bboxes[:, 3] * 0.5))
    return bboxes
