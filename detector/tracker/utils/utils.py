
import jittor as jt
from jittor import init
from jittor import nn
import glob
import random
import time
import os
import os.path as osp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import platform
if (platform.system() != 'Windows'):
    from detector.nms import nms_wrapper
torch.set_printoptions(linewidth=1320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})

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

def scale_coords(img_size, coords, img0_shape):
    coords = coords.clone()
    gain_w = (float(img_size[0]) / img0_shape[0])
    gain_h = (float(img_size[1]) / img0_shape[1])
    gain = min(gain_w, gain_h)
    pad_x = ((img_size[0] - (img0_shape[0] * gain)) / 2)
    pad_y = ((img_size[1] - (img0_shape[1] * gain)) / 2)
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
    tconf = jt.full((nB, nA, nGh, nGw), 0).long()
    tcls = jt.full((nB, nA, nGh, nGw, nC), 0).long()
    tid = jt.full((nB, nA, nGh, nGw, 1), -1).long()
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
    tconf = jt.full((nB, nA, nGh, nGw), 0).long()
    tid = jt.full((nB, nA, nGh, nGw, 1), -1).long()
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

def soft_nms(dets, sigma=0.5, Nt=0.3, threshold=0.05, method=1):
    keep = cpu_soft_nms(np.ascontiguousarray(dets, dtype=np.float32), np.float32(sigma), np.float32(Nt), np.float32(threshold), np.uint8(method))
    return keep

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4, method=(- 1)):
    "\n    Removes detections with lower object confidence score than 'conf_thres'\n    Non-Maximum Suppression to further filter detections.\n    Returns detections with shape:\n        (x1, y1, x2, y2, object_conf, class_score, class_pred)\n    "
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
        if ((method == (- 1)) and (platform.system() != 'Windows')):
            nms_op = getattr(nms_wrapper, 'nms')
            (_, nms_indices) = nms_op(pred[:, :5], nms_thres)
        else:
            dets = pred[:, :5].clone().numpy()
            nms_indices = soft_nms(dets, Nt=nms_thres, method=method)
        det_max = pred[nms_indices]
        if (len(det_max) > 0):
            output[image_i] = (det_max if (output[image_i] is None) else jt.contrib.concat((output[image_i], det_max)))
    return output

def return_torch_unique_index(u, uv):
    n = uv.shape[1]
    first_unique = jt.zeros(n).long()
    for j in range(n):
        first_unique[j] = (uv[:, j:(j + 1)] == u).all(0).nonzero()[0]
    return first_unique

def strip_optimizer_from_checkpoint(filename='weights/best.pt'):
    a = jt.load(filename)
    a['optimizer'] = []
    jt.save(a, filename.replace('.pt', '_lite.pt'))

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
