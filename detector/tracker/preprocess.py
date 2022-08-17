
import jittor as jt
from jittor import init
from __future__ import division
from jittor import nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
try:
    from util import count_parameters as count
    from util import convert2cpu as cpu
except ImportError:
    from yolo.util import count_parameters as count
    from yolo.util import convert2cpu as cpu
from PIL import Image, ImageDraw

def letterbox_image(img, img_size=(1088, 608), color=(127.5, 127.5, 127.5)):
    height = img_size[1]
    width = img_size[0]
    shape = img.shape[:2]
    ratio = min((float(height) / shape[0]), (float(width) / shape[1]))
    new_shape = (round((shape[1] * ratio)), round((shape[0] * ratio)))
    dw = ((width - new_shape[0]) / 2)
    dh = ((height - new_shape[1]) / 2)
    (top, bottom) = (round((dh - 0.1)), round((dh + 0.1)))
    (left, right) = (round((dw - 0.1)), round((dw + 0.1)))
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img

def prep_image(img, img_size=(1088, 608)):
    '\n    Prepare image for inputting to the neural network.\n\n    Returns a Variable\n    '
    orig_im = cv2.imread(img)
    dim = (orig_im.shape[1], orig_im.shape[0])
    img = letterbox_image(orig_im, img_size)
    img_ = img[:, :, ::(- 1)].transpose((2, 0, 1)).copy()
    img_ = jt.array(img_).float().div(255.0).unsqueeze(0)
    return (img_, orig_im, dim)

def prep_frame(img, img_size=(1088, 608)):
    '\n    Prepare image for inputting to the neural network.\n\n    Returns a Variable\n    '
    orig_im = img
    dim = (orig_im.shape[1], orig_im.shape[0])
    img = letterbox_image(orig_im, img_size)
    img_ = img[:, :, ::(- 1)].transpose((2, 0, 1)).copy()
    img_ = jt.array(img_).float().div(255.0).unsqueeze(0)
    return (img_, orig_im, dim)
