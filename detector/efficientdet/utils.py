
import jittor as jt
from jittor import init
from jittor import nn
import os
import cv2
import numpy as np

class AverageMeter():
    'Computes and stores the average and current value'

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += (val * n)
        self.count += n
        self.avg = (self.sum / self.count)

def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if (not os.path.exists(outdir)):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = ((outdir + '-') + str(count))
        while os.path.exists(outdir_inc):
            count = (count + 1)
            outdir_inc = ((outdir + '-') + str(count))
            assert (count < 100)
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir

def unique(tensor):
    tensor_np = tensor.numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = jt.array(unique_np)
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def letterbox_image(img, inp_dim):
    'resize image with unchanged aspect ratio using padding'
    (img_w, img_h) = (img.shape[1], img.shape[0])
    (w, h) = inp_dim
    new_w = int((img_w * min((w / img_w), (h / img_h))))
    new_h = int((img_h * min((w / img_w), (h / img_h))))
    resized_image = cv2.resize(img, (new_w, new_h))
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 0)
    canvas[0:new_h, 0:new_w, :] = resized_image
    return canvas

def prep_image(img, inp_dim):
    '\n    Prepare image for inputting to the neural network.\n\n    Returns a Variable\n    '
    orig_im = cv2.imread(img)
    dim = (orig_im.shape[1], orig_im.shape[0])
    img = letterbox_image(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::(- 1)].transpose((2, 0, 1)).copy()
    img_ = jt.array(img_).float().unsqueeze(0)
    mean = jt.array([x for x in (0.485, 0.456, 0.406)]).float().view((1, 3, 1, 1))
    std = jt.array([x for x in (0.229, 0.224, 0.225)]).float().view((1, 3, 1, 1))
    img_ = img_.div_(255).sub_(mean).div_(std)
    return (img_, orig_im, dim)

def prep_frame(img, inp_dim):
    '\n    Prepare image for inputting to the neural network.\n\n    Returns a Variable\n    '
    orig_im = img
    dim = (orig_im.shape[1], orig_im.shape[0])
    img = letterbox_image(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::(- 1)].transpose((2, 0, 1)).copy()
    img_ = jt.array(img_).float().unsqueeze(0)
    mean = jt.array([x for x in (0.485, 0.456, 0.406)]).float().view((1, 3, 1, 1))
    std = jt.array([x for x in (0.229, 0.224, 0.225)]).float().view((1, 3, 1, 1))
    img_ = img_.div_(255).sub_(mean).div_(std)
    return (img_, orig_im, dim)

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
