from __future__ import division
import jittor as jt
from jittor import init
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

def letterbox_image(img, inp_dim):
    'resize image with unchanged aspect ratio using padding'
    (img_w, img_h) = (img.shape[1], img.shape[0])
    (w, h) = inp_dim
    new_w = int((img_w * min((w / img_w), (h / img_h))))
    new_h = int((img_h * min((w / img_w), (h / img_h))))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[((h - new_h) // 2):(((h - new_h) // 2) + new_h), ((w - new_w) // 2):(((w - new_w) // 2) + new_w), :] = resized_image
    return canvas

def prep_image(img, inp_dim):
    '\n    Prepare image for inputting to the neural network.\n\n    Returns a Variable\n    '
    orig_im = cv2.imread(img)
    dim = (orig_im.shape[1], orig_im.shape[0])
    img = letterbox_image(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::(- 1)].transpose((2, 0, 1)).copy()
    img_ = (jt.array(img_).float() / 255.0).unsqueeze(0)
    return (img_, orig_im, dim)

def prep_frame(img, inp_dim):
    '\n    Prepare image for inputting to the neural network.\n\n    Returns a Variable\n    '
    orig_im = img
    dim = (orig_im.shape[1], orig_im.shape[0])
    img = letterbox_image(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::(- 1)].transpose((2, 0, 1)).copy()
    img_ = (jt.array(img_).float() / 255.0).unsqueeze(0)
    return (img_, orig_im, dim)

def prep_image_pil(img, network_dim):
    orig_im = Image.open(img)
    img = orig_im.convert('RGB')
    dim = img.size
    img = img.resize(network_dim)
    img = jt.ByteTensor(jt.ByteStorage.from_buffer(img.tobytes()))
    img = img.view((*network_dim, 3)).transpose(0, 1).transpose(0, 2)
    img = img.view((1, 3, *network_dim))
    img = img.float().div(255.0)
    return (img, orig_im, dim)

def inp_to_image(inp):
    inp = inp.jt.squeeze(-1)
    inp = (inp * 255)
    try:
        inp = inp.data.numpy()
    except RuntimeError:
        inp = inp.numpy()
    inp = inp.transpose(1, 2, 0)
    inp = inp[:, :, ::(- 1)]
    return inp
