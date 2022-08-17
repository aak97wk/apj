
import jittor as jt
from jittor import init
from jittor import nn
import glob
import random
import time
import os
import os.path as osp
import cv2
import warnings
from functools import partial
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pickle

def mkdir_if_missing(d):
    if (not osp.exists(d)):
        os.makedirs(d)

def float3(x):
    return float(format(x, '.3f'))

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_classes(path):
    "\n    Loads class labels at 'path'\n    "
    fp = open(path, 'r')
    names = fp.read().split('\n')
    return list(filter(None, names))

def model_info(model):
    n_p = sum((x.numel() for x in model.parameters()))
    n_g = sum((x.numel() for x in model.parameters() if x.requires_grad))
    print(('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma')))
    for (i, (name, p)) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print(('%5g %50s %9s %12g %20s %12.3g %12.3g' % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std())))
    print(('Model Summary: %g layers, %g parameters, %g gradients\n' % ((i + 1), n_p, n_g)))

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = (line_thickness or (round((0.0004 * max(img.shape[0:2]))) + 1))
    color = (color or [random.randint(0, 255) for _ in range(3)])
    (c1, c2) = ((int(x[0]), int(x[1])), (int(x[2]), int(x[3])))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max((tl - 1), 1)
        t_size = cv2.getTextSize(label, 0, fontScale=(tl / 3), thickness=tf)[0]
        c2 = ((c1[0] + t_size[0]), ((c1[1] - t_size[1]) - 3))
        cv2.rectangle(img, c1, c2, color, (- 1))
        cv2.putText(img, label, (c1[0], (c1[1] - 2)), 0, (tl / 3), [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != (- 1)):
        init.gauss_(m.weight.data, mean=0.0, std=0.03)
    elif (classname.find('BatchNorm2d') != (- 1)):
        init.gauss_(m.weight.data, mean=1.0, std=0.03)
        init.constant_(m.bias.data, value=0.0)

def xyxy2xywh(x):
    y = (jt.zeros(x.shape) if (x.dtype is jt.float32) else np.zeros(x.shape))
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2)
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2)
    y[:, 2] = (x[:, 2] - x[:, 0])
    y[:, 3] = (x[:, 3] - x[:, 1])
    return y

def xywh2xyxy(x):
    y = (jt.zeros(x.shape) if (x.dtype is jt.float32) else np.zeros(x.shape))
    y[:, 0] = (x[:, 0] - (x[:, 2] / 2))
    y[:, 1] = (x[:, 1] - (x[:, 3] / 2))
    y[:, 2] = (x[:, 0] + (x[:, 2] / 2))
    y[:, 3] = (x[:, 1] + (x[:, 3] / 2))
    return y

def x1y1x2y2_to_xywh(det):
    (x1, y1, x2, y2) = det
    (w, h) = ((int(x2) - int(x1)), (int(y2) - int(y1)))
    return [x1, y1, w, h]

def xywh_to_x1y1x2y2(det):
    (x1, y1, w, h) = det
    (x2, y2) = ((x1 + w), (y1 + h))
    return [x1, y1, x2, y2]

def expandBbox(xywh, width, height):
    scale = 0.05
    if isinstance(xywh, (tuple, list)):
        if (not (len(xywh) == 4)):
            raise IndexError('Bounding boxes must have 4 elements, given {}'.format(len(xywh)))
        center_x = (xywh[0] + (xywh[2] / 2))
        center_y = (xywh[1] + (xywh[3] / 2))
        (img_width, img_height) = ((xywh[2] + (scale * width)), (xywh[3] + (scale * height)))
        x1 = np.minimum((width - 1), np.maximum(0, (center_x - (img_width / 2))))
        y1 = np.minimum((height - 1), np.maximum(0, (center_y - (img_height / 2))))
        x2 = np.minimum((width - 1), np.maximum(0, (center_x + (img_width / 2))))
        y2 = np.minimum((height - 1), np.maximum(0, (center_y + (img_height / 2))))
        return (x1, y1, x2, y2)
    else:
        raise TypeError('Expect input xywh a list or tuple, given {}'.format(type(xywh)))

def bbox_clip_xyxy(xyxy, width, height):
    'Clip bounding box with format (xmin, ymin, xmax, ymax) to specified boundary.\n\n    All bounding boxes will be clipped to the new region `(0, 0, width, height)`.\n\n    Parameters\n    ----------\n    xyxy : list, tuple or numpy.ndarray\n        The bbox in format (xmin, ymin, xmax, ymax).\n        If numpy.ndarray is provided, we expect multiple bounding boxes with\n        shape `(N, 4)`.\n    width : int or float\n        Boundary width.\n    height : int or float\n        Boundary height.\n\n    Returns\n    -------\n    type\n        Description of returned object.\n\n    '
    if isinstance(xyxy, (tuple, list)):
        if (not (len(xyxy) == 4)):
            raise IndexError('Bounding boxes must have 4 elements, given {}'.format(len(xyxy)))
        x1 = np.minimum((width - 1), np.maximum(0, xyxy[0]))
        y1 = np.minimum((height - 1), np.maximum(0, xyxy[1]))
        x2 = np.minimum((width - 1), np.maximum(0, xyxy[2]))
        y2 = np.minimum((height - 1), np.maximum(0, xyxy[3]))
        return (x1, y1, x2, y2)
    elif isinstance(xyxy, np.ndarray):
        if (not ((xyxy.size % 4) == 0)):
            raise IndexError('Bounding boxes must have n * 4 elements, given {}'.format(xyxy.shape))
        x1 = np.minimum((width - 1), np.maximum(0, xyxy[:, 0]))
        y1 = np.minimum((height - 1), np.maximum(0, xyxy[:, 1]))
        x2 = np.minimum((width - 1), np.maximum(0, xyxy[:, 2]))
        y2 = np.minimum((height - 1), np.maximum(0, xyxy[:, 3]))
        return np.hstack((x1, y1, x2, y2))
    else:
        raise TypeError('Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xyxy)))

def scale_coords(img_size, coords, img0_shape):
    gain_w = (float(img_size[0]) / img0_shape[1])
    gain_h = (float(img_size[1]) / img0_shape[0])
    gain = min(gain_w, gain_h)
    pad_x = ((img_size[0] - (img0_shape[1] * gain)) / 2)
    pad_y = ((img_size[1] - (img0_shape[0] * gain)) / 2)
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, 0:4] /= gain
    coords[:, :4] = jt.clamp(coords[:, :4], min_v=0)
    return coords

def ap_per_class(tp, conf, pred_cls, target_cls):
    ' Compute the average precision, given the recall and precision curves.\n    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.\n    # Arguments\n        tp:    True positives (list).\n        conf:  Objectness value from 0-1 (list).\n        pred_cls: Predicted object classes (list).\n        target_cls: True object classes (list).\n    # Returns\n        The average precision as computed in py-faster-rcnn.\n    '
    (tp, conf, pred_cls, target_cls) = (np.array(tp), np.array(conf), np.array(pred_cls), np.array(target_cls))
    i = np.argsort((- conf))
    (tp, conf, pred_cls) = (tp[i], conf[i], pred_cls[i])
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))
    (ap, p, r) = ([], [], [])
    for c in unique_classes:
        i = (pred_cls == c)
        n_gt = sum((target_cls == c))
        n_p = sum(i)
        if ((n_p == 0) and (n_gt == 0)):
            continue
        elif ((n_p == 0) or (n_gt == 0)):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            fpc = np.cumsum((1 - tp[i]))
            tpc = np.cumsum(tp[i])
            recall_curve = (tpc / (n_gt + 1e-16))
            r.append((tpc[(- 1)] / (n_gt + 1e-16)))
            precision_curve = (tpc / (tpc + fpc))
            p.append((tpc[(- 1)] / (tpc[(- 1)] + fpc[(- 1)])))
            ap.append(compute_ap(recall_curve, precision_curve))
    return (np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p))

def compute_ap(recall, precision):
    ' Compute the average precision, given the recall and precision curves.\n    Code originally from https://github.com/rbgirshick/py-faster-rcnn.\n    # Arguments\n        recall:    The recall curve (list).\n        precision: The precision curve (list).\n    # Returns\n        The average precision as computed in py-faster-rcnn.\n    '
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range((mpre.size - 1), 0, (- 1)):
        mpre[(i - 1)] = np.maximum(mpre[(i - 1)], mpre[i])
    i = np.where((mrec[1:] != mrec[:(- 1)]))[0]
    ap = np.sum(((mrec[(i + 1)] - mrec[i]) * mpre[(i + 1)]))
    return ap

def bbox_iou(box1, box2, x1y1x2y2=False):
    '\n    Returns the IoU of two bounding boxes\n    '
    (N, M) = (len(box1), len(box2))
    if x1y1x2y2:
        (b1_x1, b1_y1, b1_x2, b1_y2) = (box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3])
        (b2_x1, b2_y1, b2_x2, b2_y2) = (box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3])
    else:
        (b1_x1, b1_x2) = ((box1[:, 0] - (box1[:, 2] / 2)), (box1[:, 0] + (box1[:, 2] / 2)))
        (b1_y1, b1_y2) = ((box1[:, 1] - (box1[:, 3] / 2)), (box1[:, 1] + (box1[:, 3] / 2)))
        (b2_x1, b2_x2) = ((box2[:, 0] - (box2[:, 2] / 2)), (box2[:, 0] + (box2[:, 2] / 2)))
        (b2_y1, b2_y2) = ((box2[:, 1] - (box2[:, 3] / 2)), (box2[:, 1] + (box2[:, 3] / 2)))
    inter_rect_x1 = jt.max(b1_x1.unsqueeze(1), dim=b2_x1)
    inter_rect_y1 = jt.max(b1_y1.unsqueeze(1), dim=b2_y1)
    inter_rect_x2 = jt.min(b1_x2.unsqueeze(1), dim=b2_x2)
    inter_rect_y2 = jt.min(b1_y2.unsqueeze(1), dim=b2_y2)
    inter_area = (jt.clamp((inter_rect_x2 - inter_rect_x1), min_v=0) * jt.clamp((inter_rect_y2 - inter_rect_y1), min_v=0))
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).view(((- 1), 1)).expand(N, M)
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).view((1, (- 1))).expand(N, M)
    return (inter_area / (((b1_area + b2_area) - inter_area) + 1e-16))

def build_targets_max(target, anchor_wh, nA, nC, nGh, nGw):
    '\n    returns nT, nCorrect, tx, ty, tw, th, tconf, tcls\n    '
    nB = len(target)
    txy = jt.zeros((nB, nA, nGh, nGw, 2))
    twh = jt.zeros((nB, nA, nGh, nGw, 2))
    tconf = jt.full((nB, nA, nGh, nGw), 0).int()
    tcls = jt.full((nB, nA, nGh, nGw, nC), 0).int()
    tid = jt.full((nB, nA, nGh, nGw), 1).int()
    for b in range(nB):
        t = target[b]
        t_id = t[:, 1].clone().long()
        t = t[:, [0, 2, 3, 4, 5]]
        nTb = len(t)
        if (nTb == 0):
            continue
        (gxy, gwh) = (t[:, 1:3].clone(), t[:, 3:5].clone())
        gxy[:, 0] = (gxy[:, 0] * nGw)
        gxy[:, 1] = (gxy[:, 1] * nGh)
        gwh[:, 0] = (gwh[:, 0] * nGw)
        gwh[:, 1] = (gwh[:, 1] * nGh)
        gi = jt.clamp(gxy[:, 0], min=0, max=(nGw - 1)).long()
        gj = jt.clamp(gxy[:, 1], min=0, max=(nGh - 1)).long()
        box1 = gwh
        box2 = anchor_wh.unsqueeze(1)
        inter_area = jt.min(box1, box2).prod(dim=2)
        iou = (inter_area / (((box1.prod(dim=1) + box2.prod(dim=2)) - inter_area) + 1e-16))
        (iou_best, a) = iou.max(dim=0)
        if (nTb > 1):
            (iou_order, _) = jt.argsort((- iou_best))
            u = jt.stack((gi, gj, a), dim=0)[:, iou_order]
            first_unique = return_torch_unique_index(u, jt.unique(u, dim=1))
            i = iou_order[first_unique]
            i = i[(iou_best[i] > 0.6)]
            if (len(i) == 0):
                continue
            (a, gj, gi, t) = (a[i], gj[i], gi[i], t[i])
            t_id = t_id[i]
            if (len(t.shape) == 1):
                t = t.view((1, 5))
        elif (iou_best < 0.6):
            continue
        (tc, gxy, gwh) = (t[:, 0].long(), t[:, 1:3].clone(), t[:, 3:5].clone())
        gxy[:, 0] = (gxy[:, 0] * nGw)
        gxy[:, 1] = (gxy[:, 1] * nGh)
        gwh[:, 0] = (gwh[:, 0] * nGw)
        gwh[:, 1] = (gwh[:, 1] * nGh)
        txy[(b, a, gj, gi)] = (gxy - gxy.floor())
        twh[(b, a, gj, gi)] = jt.log((gwh / anchor_wh[a]))
        tcls[(b, a, gj, gi, tc)] = 1
        tconf[(b, a, gj, gi)] = 1
        tid[(b, a, gj, gi)] = t_id.unsqueeze(1)
    tbox = jt.contrib.concat([txy, twh], dim=(- 1))
    return (tconf, tbox, tid)

def build_targets_thres(target, anchor_wh, nA, nC, nGh, nGw):
    ID_THRESH = 0.5
    FG_THRESH = 0.5
    BG_THRESH = 0.4
    nB = len(target)
    assert (len(anchor_wh) == nA)
    tbox = jt.zeros((nB, nA, nGh, nGw, 4))
    tconf = jt.full((nB, nA, nGh, nGw), 0).int()
    tid = jt.full((nB, nA, nGh, nGw, 1), -1).int()
    for b in range(nB):
        t = target[b]
        t_id = t[:, 1].clone().long()
        t = t[:, [0, 2, 3, 4, 5]]
        nTb = len(t)
        if (nTb == 0):
            continue
        (gxy, gwh) = (t[:, 1:3].clone(), t[:, 3:5].clone())
        gxy[:, 0] = (gxy[:, 0] * nGw)
        gxy[:, 1] = (gxy[:, 1] * nGh)
        gwh[:, 0] = (gwh[:, 0] * nGw)
        gwh[:, 1] = (gwh[:, 1] * nGh)
        gxy[:, 0] = jt.clamp(gxy[:, 0], min_v=0, max_v=(nGw - 1))
        gxy[:, 1] = jt.clamp(gxy[:, 1], min_v=0, max_v=(nGh - 1))
        gt_boxes = jt.contrib.concat([gxy, gwh], dim=1)
        anchor_mesh = generate_anchor(nGh, nGw, anchor_wh)
        anchor_list = anchor_mesh.permute(0, 2, 3, 1).view(((- 1), 4))
        iou_pdist = bbox_iou(anchor_list, gt_boxes)
        (iou_max, max_gt_index) = jt.max(iou_pdist, dim=1)
        iou_map = iou_max.view((nA, nGh, nGw))
        gt_index_map = max_gt_index.view((nA, nGh, nGw))
        id_index = (iou_map > ID_THRESH)
        fg_index = (iou_map > FG_THRESH)
        bg_index = (iou_map < BG_THRESH)
        ign_index = ((iou_map < FG_THRESH) * (iou_map > BG_THRESH))
        tconf[b][fg_index] = 1
        tconf[b][bg_index] = 0
        tconf[b][ign_index] = (- 1)
        gt_index = gt_index_map[fg_index]
        gt_box_list = gt_boxes[gt_index]
        gt_id_list = t_id[gt_index_map[id_index]]
        if (jt.sum(fg_index) > 0):
            tid[b][id_index] = gt_id_list.unsqueeze(1)
            fg_anchor_list = anchor_list.view((nA, nGh, nGw, 4))[fg_index]
            delta_target = encode_delta(gt_box_list, fg_anchor_list)
            tbox[b][fg_index] = delta_target
    return (tconf, tbox, tid)

def generate_anchor(nGh, nGw, anchor_wh):
    nA = len(anchor_wh)
    (yy, xx) = jt.meshgrid((jt.arange(nGh), jt.arange(nGw)))
    # (xx, yy) = (xx, yy.cuda())
    mesh = jt.stack([xx, yy], dim=0)
    mesh = mesh.unsqueeze(0).repeat(nA, 1, 1, 1).float()
    anchor_offset_mesh = anchor_wh.unsqueeze((- 1)).unsqueeze((- 1)).repeat(1, 1, nGh, nGw)
    anchor_mesh = jt.contrib.concat([mesh, anchor_offset_mesh], dim=1)
    return anchor_mesh

def encode_delta(gt_box_list, fg_anchor_list):
    (px, py, pw, ph) = (fg_anchor_list[:, 0], fg_anchor_list[:, 1], fg_anchor_list[:, 2], fg_anchor_list[:, 3])
    (gx, gy, gw, gh) = (gt_box_list[:, 0], gt_box_list[:, 1], gt_box_list[:, 2], gt_box_list[:, 3])
    dx = ((gx - px) / pw)
    dy = ((gy - py) / ph)
    dw = jt.log((gw / pw))
    dh = jt.log((gh / ph))
    return jt.stack([dx, dy, dw, dh], dim=1)

def decode_delta(delta, fg_anchor_list):
    (px, py, pw, ph) = (fg_anchor_list[:, 0], fg_anchor_list[:, 1], fg_anchor_list[:, 2], fg_anchor_list[:, 3])
    (dx, dy, dw, dh) = (delta[:, 0], delta[:, 1], delta[:, 2], delta[:, 3])
    gx = ((pw * dx) + px)
    gy = ((ph * dy) + py)
    gw = (pw * jt.exp(dw))
    gh = (ph * jt.exp(dh))
    return jt.stack([gx, gy, gw, gh], dim=1)

def decode_delta_map(delta_map, anchors):
    '\n    :param: delta_map, shape (nB, nA, nGh, nGw, 4)\n    :param: anchors, shape (nA,4)\n    '
    (nB, nA, nGh, nGw, _) = delta_map.shape
    anchor_mesh = generate_anchor(nGh, nGw, anchors)
    anchor_mesh = anchor_mesh.permute((0, 2, 3, 1))
    anchor_mesh = anchor_mesh.unsqueeze(0).repeat(nB, 1, 1, 1, 1)
    pred_list = decode_delta(delta_map.view(((- 1), 4)), anchor_mesh.view(((- 1), 4)))
    pred_map = pred_list.view((nB, nA, nGh, nGw, 4))
    return pred_map

def pooling_nms(heatmap, kernel=1):
    pad = ((kernel - 1) // 2)
    hmax = nn.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return (keep * heatmap)

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4, method='standard'):
    "\n    Removes detections with lower object confidence score than 'conf_thres'\n    Non-Maximum Suppression to further filter detections.\n    Returns detections with shape:\n        (x1, y1, x2, y2, object_conf, class_score, class_pred)\n    Args:\n        prediction,\n        conf_thres,\n        nms_thres,\n        method = 'standard' or 'fast'\n    "
    output = [None for _ in range(len(prediction))]
    for (image_i, pred) in enumerate(prediction):
        v = (pred[:, 4] > conf_thres)
        v = v.nonzero().squeeze(-1)
        if (len(v.shape) == 0):
            v = v.unsqueeze(0)
        pred = pred[v]
        nP = pred.shape[0]
        if (not nP):
            continue
        pred[:, :4] = xywh2xyxy(pred[:, :4])
        if (method == 'standard'):
            nms_indices = nms(pred[:, :4], pred[:, 4], nms_thres)
        elif (method == 'fast'):
            nms_indices = fast_nms(pred[:, :4], pred[:, 4], iou_thres=nms_thres, conf_thres=conf_thres)
        else:
            raise ValueError('Invalid NMS type!')
        det_max = pred[nms_indices]
        if (len(det_max) > 0):
            output[image_i] = (det_max if (output[image_i] is None) else jt.contrib.concat((output[image_i], det_max)))
    return output

def fast_nms(boxes, scores, iou_thres: float=0.5, top_k: int=200, second_threshold: bool=False, conf_thres: float=0.5):
    '\n    Vectorized, approximated, fast NMS, adopted from YOLACT:\n    https://github.com/dbolya/yolact/blob/master/layers/functions/detection.py\n    The original version is for multi-class NMS, here we simplify the code for single-class NMS\n    '
    (scores, idx) = scores.sort(0, descending=True)
    idx = idx[:top_k]
    scores = scores[:top_k]
    num_dets = idx.shape
    boxes = boxes[idx, :]
    iou = jaccard(boxes, boxes)
    iou.triu_(diagonal=1)
    (iou_max, _) = iou.max(dim=0)
    keep = (iou_max <= iou_thres)
    if second_threshold:
        keep *= (scores > conf_thres)
    return idx[keep]

# @torch.jit.script
def intersect(box_a, box_b):
    ' We resize both tensors to [A,B,2] without new malloc:\n    [A,2] -> [A,1,2] -> [A,B,2]\n    [B,2] -> [1,B,2] -> [A,B,2]\n    Then we compute the area of intersect between box_a and box_b.\n    Args:\n      box_a: (tensor) bounding boxes, Shape: [n,A,4].\n      box_b: (tensor) bounding boxes, Shape: [n,B,4].\n    Return:\n      (tensor) intersection area, Shape: [n,A,B].\n    '
    n = box_a.shape[0]
    A = box_a.shape[1]
    B = box_b.shape[1]
    max_xy = jt.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2), dim=box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = jt.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2), dim=box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    inter = jt.clamp((max_xy - min_xy), min_v=0)
    return (inter[:, :, :, 0] * inter[:, :, :, 1])

def jaccard(box_a, box_b, iscrowd: bool=False):
    'Compute the jaccard overlap of two sets of boxes.  The jaccard overlap\n    is simply the intersection over union of two boxes.  Here we operate on\n    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.\n    E.g.:\n        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)\n    Args:\n        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]\n        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]\n    Return:\n        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]\n    '
    use_batch = True
    if (box_a.ndim == 2):
        use_batch = False
        box_a = box_a[(None, ...)]
        box_b = box_b[(None, ...)]
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)
    union = ((area_a + area_b) - inter)
    out = ((inter / area_a) if iscrowd else (inter / union))
    return (out if use_batch else out.squeeze(0))

def return_torch_unique_index(u, uv):
    n = uv.shape[1]
    first_unique = jtzeros(n).long()
    for j in range(n):
        first_unique[j] = (uv[:, j:(j + 1)] == u).all(0).nonzero()[0]
    return first_unique

def strip_optimizer_from_checkpoint(filename='weights/best.pt'):
    a = jt.load(filename)
    a['optimizer'] = []
    jtsave(a, filename.replace('.pt', '_lite.pt'))

def plot_results():
    plt.figure(figsize=(14, 7))
    s = ['X + Y', 'Width + Height', 'Confidence', 'Classification', 'Total Loss', 'mAP', 'Recall', 'Precision']
    files = sorted(glob.glob('results*.txt'))
    for f in files:
        results = np.loadtxt(f, usecols=[2, 3, 4, 5, 6, 9, 10, 11]).T
        x = range(1, results.shape[1])
        for i in range(8):
            plt.subplot(2, 4, (i + 1))
            plt.plot(x, results[(i, x)], marker='.', label=f)
            plt.title(s[i])
            if (i == 0):
                plt.legend()

def load_checkpoint(fpath):
    "Loads checkpoint.\n\n    ``UnicodeDecodeError`` can be well handled, which means\n    python2-saved files can be read from python3.\n\n    Args:\n        fpath (str): path to checkpoint.\n\n    Returns:\n        dict\n\n    Examples::\n        >>> from torchreid.utils import load_checkpoint\n        >>> fpath = 'log/my_model/model.pth.tar-10'\n        >>> checkpoint = load_checkpoint(fpath)\n    "
    if (fpath is None):
        raise ValueError('File path is None')
    if (not osp.exists(fpath)):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = (None if jt.has_cuda else 'cpu')
    try:
        checkpoint = jt.load(fpath)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding='latin1')
        pickle.Unpickler = partial(pickle.Unpickler, encoding='latin1')
        checkpoint = jt.load(fpath)
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint

def load_pretrained_weights(model, weight_path):
    'Loads pretrianed weights to model.\n\n    Features::\n        - Incompatible layers (unmatched in name or size) will be ignored.\n        - Can automatically deal with keys containing "module.".\n\n    Args:\n        model (nn.Module): network model.\n        weight_path (str): path to pretrained weights.\n\n    Examples::\n        >>> from torchreid.utils import load_pretrained_weights\n        >>> weight_path = \'log/my_model/model-best.pth.tar\'\n        >>> load_pretrained_weights(model, weight_path)\n    '
    checkpoint = load_checkpoint(weight_path)
    if ('state_dict' in checkpoint):
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    (matched_layers, discarded_layers) = ([], [])
    "\n    print('keys of loaded model:')\n    for k, v in state_dict.items():\n        print(k)\n    print('keys of model archs:')\n    for k, v in model_dict.items():\n        print(k)\n    "
    for (k, v) in state_dict.items():
        if (not k.startswith('module.')):
            k = ('module.' + k)
        if ((k in model_dict) and (model_dict[k].shape == v.shape)):
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_parameters(model_dict)
    if (len(matched_layers) == 0):
        warnings.warn('The pretrained weights "{}" cannot be loaded, please check the key names manually (** ignored and continue **)'.format(weight_path))
    else:
        print('loading reid model from {}...'.format(weight_path))
        "\n        if len(discarded_layers) > 0:\n            print(\n                '** The following layers are discarded '\n                'due to unmatched keys or layer size: {}'.\n                format(discarded_layers)\n            )\n        "
