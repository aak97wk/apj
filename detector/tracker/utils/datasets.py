
import jittor as jt
from jittor import init
from jittor import nn
import glob
import math
import os
import os.path as osp
import random
import time
from collections import OrderedDict
import cv2
import numpy as np
from utils.utils import xyxy2xywh

class LoadImages():

    def __init__(self, path, img_size=(1088, 608)):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob(('%s/*.*' % path)))
            self.files = list(filter((lambda x: (os.path.splitext(x)[1].lower() in image_format)), self.files))
        elif os.path.isfile(path):
            self.files = [path]
        self.nF = len(self.files)
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0
        assert (self.nF > 0), ('No images found in ' + path)

    def __iter__(self):
        self.count = (- 1)
        return self

    def __next__(self):
        self.count += 1
        if (self.count == self.nF):
            raise StopIteration
        img_path = self.files[self.count]
        img0 = cv2.imread(img_path)
        assert (img0 is not None), ('Failed to load ' + img_path)
        (img, _, _, _) = letterbox(img0, height=self.height, width=self.width)
        img = img[:, :, ::(- 1)].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        return (img_path, img, img0)

    def __getitem__(self, idx):
        idx = (idx % self.nF)
        img_path = self.files[idx]
        img0 = cv2.imread(img_path)
        assert (img0 is not None), ('Failed to load ' + img_path)
        (img, _, _, _) = letterbox(img0, height=self.height, width=self.width)
        img = img[:, :, ::(- 1)].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        return (img_path, img, img0)

    def __len__(self):
        return self.nF

class LoadVideo():

    def __init__(self, path, img_size=(1088, 608)):
        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0
        (self.w, self.h) = self.get_size(self.vw, self.vh, self.width, self.height)
        print('Lenth of the video: {:d} frames'.format(self.vn))

    def get_size(self, vw, vh, dw, dh):
        (wa, ha) = ((float(dw) / vw), (float(dh) / vh))
        a = min(wa, ha)
        return (int((vw * a)), int((vh * a)))

    def __iter__(self):
        self.count = (- 1)
        return self

    def __next__(self):
        self.count += 1
        if (self.count == len(self)):
            raise StopIteration
        (res, img0) = self.cap.read()
        assert (img0 is not None), 'Failed to load frame {:d}'.format(self.count)
        img0 = cv2.resize(img0, (self.w, self.h))
        (img, _, _, _) = letterbox(img0, height=self.height, width=self.width)
        img = img[:, :, ::(- 1)].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        return (self.count, img, img0)

    def __len__(self):
        return self.vn

class LoadImagesAndLabels():

    def __init__(self, path, img_size=(1088, 608), augment=False, transforms=None):
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter((lambda x: (len(x) > 0)), self.img_files))
        self.label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt') for x in self.img_files]
        self.nF = len(self.img_files)
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

    def __getitem__(self, files_index):
        img_path = self.img_files[files_index]
        label_path = self.label_files[files_index]
        return self.get_data(img_path, label_path)

    def get_data(self, img_path, label_path):
        height = self.height
        width = self.width
        img = cv2.imread(img_path)
        if (img is None):
            raise ValueError('File corrupt {}'.format(img_path))
        augment_hsv = True
        if (self.augment and augment_hsv):
            fraction = 0.5
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)
            a = ((((random.random() * 2) - 1) * fraction) + 1)
            S *= a
            if (a > 1):
                np.clip(S, a_min=0, a_max=255, out=S)
            a = ((((random.random() * 2) - 1) * fraction) + 1)
            V *= a
            if (a > 1):
                np.clip(V, a_min=0, a_max=255, out=V)
            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        (h, w, _) = img.shape
        (img, ratio, padw, padh) = letterbox(img, height=height, width=width)
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape((- 1), 6)
            labels = labels0.copy()
            labels[:, 2] = (((ratio * w) * (labels0[:, 2] - (labels0[:, 4] / 2))) + padw)
            labels[:, 3] = (((ratio * h) * (labels0[:, 3] - (labels0[:, 5] / 2))) + padh)
            labels[:, 4] = (((ratio * w) * (labels0[:, 2] + (labels0[:, 4] / 2))) + padw)
            labels[:, 5] = (((ratio * h) * (labels0[:, 3] + (labels0[:, 5] / 2))) + padh)
        else:
            labels = np.array([])
        if self.augment:
            (img, labels, M) = random_affine(img, labels, degrees=((- 5), 5), translate=(0.1, 0.1), scale=(0.5, 1.2))
        plotFlag = False
        if plotFlag:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(50, 50))
            plt.imshow(img[:, :, ::(- 1)])
            plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [2, 2, 4, 4, 2]].T, '.-')
            plt.axis('off')
            plt.savefig('test.jpg')
            time.sleep(10)
        nL = len(labels)
        if (nL > 0):
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height
        if self.augment:
            lr_flip = True
            if (lr_flip & (random.random() > 0.5)):
                img = np.fliplr(img)
                if (nL > 0):
                    labels[:, 2] = (1 - labels[:, 2])
        img = np.ascontiguousarray(img[:, :, ::(- 1)])
        if (self.transforms is not None):
            img = self.transforms(img)
        return (img, labels, img_path, (h, w))

    def __len__(self):
        return self.nF

def letterbox(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):
    shape = img.shape[:2]
    ratio = min((float(height) / shape[0]), (float(width) / shape[1]))
    new_shape = (round((shape[1] * ratio)), round((shape[0] * ratio)))
    dw = ((width - new_shape[0]) / 2)
    dh = ((height - new_shape[1]) / 2)
    (top, bottom) = (round((dh - 0.1)), round((dh + 0.1)))
    (left, right) = (round((dw - 0.1)), round((dw + 0.1)))
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return (img, ratio, dw, dh)

def random_affine(img, targets=None, degrees=((- 10), 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=((- 2), 2), borderValue=(127.5, 127.5, 127.5)):
    border = 0
    height = img.shape[0]
    width = img.shape[1]
    R = np.eye(3)
    a = ((random.random() * (degrees[1] - degrees[0])) + degrees[0])
    s = ((random.random() * (scale[1] - scale[0])) + scale[0])
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=((img.shape[1] / 2), (img.shape[0] / 2)), scale=s)
    T = np.eye(3)
    T[(0, 2)] = (((((random.random() * 2) - 1) * translate[0]) * img.shape[0]) + border)
    T[(1, 2)] = (((((random.random() * 2) - 1) * translate[1]) * img.shape[1]) + border)
    S = np.eye(3)
    S[(0, 1)] = math.tan(((((random.random() * (shear[1] - shear[0])) + shear[0]) * math.pi) / 180))
    S[(1, 0)] = math.tan(((((random.random() * (shear[1] - shear[0])) + shear[0]) * math.pi) / 180))
    M = ((S @ T) @ R)
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=borderValue)
    if (targets is not None):
        if (len(targets) > 0):
            n = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0 = ((points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1]))
            xy = np.ones(((n * 4), 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(((n * 4), 2))
            xy = (xy @ M.T)[:, :2].reshape((n, 8))
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            radians = ((a * math.pi) / 180)
            reduction = (max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5)
            x = ((xy[:, 2] + xy[:, 0]) / 2)
            y = ((xy[:, 3] + xy[:, 1]) / 2)
            w = ((xy[:, 2] - xy[:, 0]) * reduction)
            h = ((xy[:, 3] - xy[:, 1]) * reduction)
            xy = np.concatenate(((x - (w / 2)), (y - (h / 2)), (x + (w / 2)), (y + (h / 2)))).reshape(4, n).T
            np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = (xy[:, 2] - xy[:, 0])
            h = (xy[:, 3] - xy[:, 1])
            area = (w * h)
            ar = np.maximum((w / (h + 1e-16)), (h / (w + 1e-16)))
            i = ((((w > 4) & (h > 4)) & ((area / (area0 + 1e-16)) > 0.1)) & (ar < 10))
            targets = targets[i]
            targets[:, 2:6] = xy[i]
        return (imw, targets, M)
    else:
        return imw

def collate_fn(batch):
    (imgs, labels, paths, sizes) = zip(*batch)
    batch_size = len(labels)
    imgs = jt.stack(imgs, dim=0)
    max_box_len = max([l.shape[0] for l in labels])
    labels = [jt.array(l) for l in labels]
    filled_labels = jt.zeros((batch_size, max_box_len, 6))
    labels_len = jt.zeros(batch_size)
    for i in range(batch_size):
        isize = labels[i].shape[0]
        if (len(labels[i]) > 0):
            filled_labels[i, :isize, :] = labels[i]
        labels_len[i] = isize
    return (imgs, filled_labels, paths, sizes, labels_len.unsqueeze(1))

class JointDataset(LoadImagesAndLabels):

    def __init__(self, root, paths, img_size=(1088, 608), augment=False, transforms=None):
        dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        for (ds, path) in paths.items():
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter((lambda x: (len(x) > 0)), self.img_files[ds]))
            self.label_files[ds] = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt') for x in self.img_files[ds]]
        for (ds, label_paths) in self.label_files.items():
            max_index = (- 1)
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if (len(lb) < 1):
                    continue
                if (len(lb.shape) < 2):
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if (img_max > max_index):
                    max_index = img_max
            self.tid_num[ds] = (max_index + 1)
        last_index = 0
        for (i, (k, v)) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v
        self.nID = int((last_index + 1))
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms
        print(('=' * 80))
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nID)
        print('start index')
        print(self.tid_start_index)
        print(('=' * 80))

    def __getitem__(self, files_index):
        for (i, c) in enumerate(self.cds):
            if (files_index >= c):
                ds = list(self.label_files.keys())[i]
                start_index = c
        img_path = self.img_files[ds][(files_index - start_index)]
        label_path = self.label_files[ds][(files_index - start_index)]
        (imgs, labels, img_path, (h, w)) = self.get_data(img_path, label_path)
        for (i, _) in enumerate(labels):
            if (labels[(i, 1)] > (- 1)):
                labels[(i, 1)] += self.tid_start_index[ds]
        return (imgs, labels, img_path, (h, w))
