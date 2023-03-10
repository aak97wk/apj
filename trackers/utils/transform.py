
import jittor as jt
from jittor import init
from jittor import nn
import math
import random
import cv2
import numpy as np

class RandomErasing(object):
    " Randomly selects a rectangle region in an image and erases its pixels.\n        'Random Erasing Data Augmentation' by Zhong et al.\n        See https://arxiv.org/pdf/1708.04896.pdf\n    Args:\n        probability: The probability that the Random Erasing operation will be performed.\n        sl: Minimum proportion of erased area against input image.\n        sh: Maximum proportion of erased area against input image.\n        r1: Minimum aspect ratio of erased area.\n        mean: Erasing value.\n    "

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(255 * (0.49735, 0.4822, 0.4465))):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        img = np.asarray(img, dtype=np.float32).copy()
        if (random.uniform(0, 1) > self.probability):
            return img
        for attempt in range(100):
            area = (img.shape[0] * img.shape[1])
            target_area = (random.uniform(self.sl, self.sh) * area)
            aspect_ratio = random.uniform(self.r1, (1 / self.r1))
            h = int(round(math.sqrt((target_area * aspect_ratio))))
            w = int(round(math.sqrt((target_area / aspect_ratio))))
            if ((w < img.shape[1]) and (h < img.shape[0])):
                x1 = random.randint(0, (img.shape[0] - h))
                y1 = random.randint(0, (img.shape[1] - w))
                if (img.shape[2] == 3):
                    img[x1:(x1 + h), y1:(y1 + w), 0] = self.mean[0]
                    img[x1:(x1 + h), y1:(y1 + w), 1] = self.mean[1]
                    img[x1:(x1 + h), y1:(y1 + w), 2] = self.mean[2]
                else:
                    img[x1:(x1 + h), y1:(y1 + w), 0] = self.mean[0]
                return img
        return img

def to_tensor(pic):
    'Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.\n\n    See ``ToTensor`` for more details.\n\n    Args:\n        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.\n\n    Returns:\n        Tensor: Converted image.\n    '
    if isinstance(pic, np.ndarray):
        assert (len(pic.shape) in (2, 3))
        if (pic.ndim == 2):
            pic = pic[:, :, None]
        img = jt.array(pic.transpose((2, 0, 1)))
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
    if (pic.mode == 'I'):
        img = jt.array(np.array(pic, np.int32, copy=False))
    elif (pic.mode == 'I;16'):
        img = jt.array(np.array(pic, np.int16, copy=False))
    elif (pic.mode == 'F'):
        img = jt.array(np.array(pic, np.float32, copy=False))
    elif (pic.mode == '1'):
        img = (255 * jt.array(np.array(pic, np.uint8, copy=False)))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    if (pic.mode == 'YCbCr'):
        nchannel = 3
    elif (pic.mode == 'I;16'):
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view((pic.size[1], pic.size[0], nchannel))
    img = img.transpose(0, 1).transpose(0, 2)
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img

class ToTensor(object):
    'Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.\n\n    Converts a PIL Image or numpy.ndarray (H x W x C) in the range\n    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]\n    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)\n    or if the numpy.ndarray has dtype = np.uint8\n\n    In the other cases, tensors are returned without scaling.\n    '

    def __call__(self, pic):
        '\n        Args:\n            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.\n\n        Returns:\n            Tensor: Converted image.\n        '
        return to_tensor(pic)

    def __repr__(self):
        return (self.__class__.__name__ + '()')

def build_transforms(cfg, is_train=True):
    res = []
    res.append(T.ToPILImage(mode=None))
    if is_train:
        size_train = cfg['SIZE_TRAIN']
        do_flip = cfg['DO_FLIP']
        flip_prob = cfg['FLIP_PROB']
        do_pad = cfg['DO_PAD']
        padding = cfg['PADDING']
        padding_mode = cfg['PADDING_MODE']
        do_re = cfg['RE_ENABLED']
        res.append(T.Resize(size_train, interpolation=3))
        if do_flip:
            res.append(T.RandomHorizontalFlip(p=flip_prob))
        if do_pad:
            res.extend([T.Pad(padding, padding_mode=padding_mode), T.RandomCrop(size_train)])
        if do_re:
            res.append(RandomErasing())
    else:
        size_test = cfg['TEST_SIZE']
        res.append(T.Resize(size_test, interpolation=3))
    res.append(ToTensor())
    return T.Compose(res)
