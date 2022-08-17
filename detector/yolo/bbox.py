from __future__ import division
import jittor as jt
from jittor import init
from jittor import nn
import random
import numpy as np
import cv2

def confidence_filter(result, confidence):
    conf_mask = (result[:, :, 4] > confidence).float().unsqueeze(2)
    result = (result * conf_mask)
    return result

def confidence_filter_cls(result, confidence):
    max_scores = jt.max(result[:, :, 5:25], dim=2)[0]
    res = jt.contrib.concat((result, max_scores), dim=2)
    print(res.shape)
    cond_1 = (res[:, :, 4] > confidence).float()
    cond_2 = (res[:, :, 25] > 0.995).float()
    conf = (cond_1 + cond_2)
    conf = jt.clamp(conf, min_v=0.0, max_v=1.0)
    conf = conf.unsqueeze(2)
    result = (result * conf)
    return result

def get_abs_coord(box):
    (box[2], box[3]) = (abs(box[2]), abs(box[3]))
    x1 = ((box[0] - (box[2] / 2)) - 1)
    y1 = ((box[1] - (box[3] / 2)) - 1)
    x2 = ((box[0] + (box[2] / 2)) - 1)
    y2 = ((box[1] + (box[3] / 2)) - 1)
    return (x1, y1, x2, y2)

def sanity_fix(box):
    if (box[0] > box[2]):
        (box[0], box[2]) = (box[2], box[0])
    if (box[1] > box[3]):
        (box[1], box[3]) = (box[3], box[1])
    return box

def bbox_iou(box1, box2, args=None):
    '\n    Returns the IoU of two bounding boxes \n    \n    \n    '
    (b1_x1, b1_y1, b1_x2, b1_y2) = (box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3])
    (b2_x1, b2_y1, b2_x2, b2_y2) = (box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3])
    inter_rect_x1 = jt.max(b1_x1, dim=b2_x1)
    inter_rect_y1 = jt.max(b1_y1, dim=b2_y1)
    inter_rect_x2 = jt.min(b1_x2, dim=b2_x2)
    inter_rect_y2 = jt.min(b1_y2, dim=b2_y2)
    if (not args):
        inter_area = (jt.max(((inter_rect_x2 - inter_rect_x1) + 1), dim=jt.zeros(inter_rect_x2.shape)) * jt.max(((inter_rect_y2 - inter_rect_y1) + 1), dim=jt.zeros(inter_rect_x2.shape)))
    else:
        inter_area = (jt.max(((inter_rect_x2 - inter_rect_x1) + 1), dim=jt.zeros(inter_rect_x2.shape)) * jt.max(((inter_rect_y2 - inter_rect_y1) + 1), dim=jt.zeros(inter_rect_x2.shape)))
    b1_area = (((b1_x2 - b1_x1) + 1) * ((b1_y2 - b1_y1) + 1))
    b2_area = (((b2_x2 - b2_x1) + 1) * ((b2_y2 - b2_y1) + 1))
    iou = (inter_area / ((b1_area + b2_area) - inter_area))
    return iou

def pred_corner_coord(prediction):
    ind_nz = jt.nonzero(prediction[:, :, 4]).transpose(0, 1)
    box = prediction[(ind_nz[0], ind_nz[1])]
    box_a = box.new(box.shape)
    box_a[:, 0] = (box[:, 0] - (box[:, 2] / 2))
    box_a[:, 1] = (box[:, 1] - (box[:, 3] / 2))
    box_a[:, 2] = (box[:, 0] + (box[:, 2] / 2))
    box_a[:, 3] = (box[:, 1] + (box[:, 3] / 2))
    box[:, :4] = box_a[:, :4]
    prediction[(ind_nz[0], ind_nz[1])] = box
    return prediction

def write(x, batches, results, colors, classes):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[(- 1)])
    label = '{0}'.format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = (((c1[0] + t_size[0]) + 3), ((c1[1] + t_size[1]) + 4))
    cv2.rectangle(img, c1, c2, color, (- 1))
    cv2.putText(img, label, (c1[0], ((c1[1] + t_size[1]) + 4)), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img
