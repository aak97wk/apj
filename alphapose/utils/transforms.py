
import jittor as jt
from jittor import init
from jittor import nn
'Pose related transforrmation functions.'
import random
import cv2
import numpy as np

def rnd(x):
    return max(((- 2) * x), min((2 * x), (np.random.randn(1)[0] * x)))

def box_transform(bbox, sf, imgwidth, imght, train):
    'Random scaling.'
    width = (bbox[2] - bbox[0])
    ht = (bbox[3] - bbox[1])
    if train:
        scaleRate = (0.25 * np.clip((np.random.randn() * sf), (- sf), sf))
        bbox[0] = max(0, (bbox[0] - ((width * scaleRate) / 2)))
        bbox[1] = max(0, (bbox[1] - ((ht * scaleRate) / 2)))
        bbox[2] = min(imgwidth, (bbox[2] + ((width * scaleRate) / 2)))
        bbox[3] = min(imght, (bbox[3] + ((ht * scaleRate) / 2)))
    else:
        scaleRate = 0.25
        bbox[0] = max(0, (bbox[0] - ((width * scaleRate) / 2)))
        bbox[1] = max(0, (bbox[1] - ((ht * scaleRate) / 2)))
        bbox[2] = min(imgwidth, max((bbox[2] + ((width * scaleRate) / 2)), (bbox[0] + 5)))
        bbox[3] = min(imght, max((bbox[3] + ((ht * scaleRate) / 2)), (bbox[1] + 5)))
    return bbox

def addDPG(bbox, imgwidth, imght):
    'Add dpg for data augmentation, including random crop and random sample.'
    PatchScale = random.uniform(0, 1)
    width = (bbox[2] - bbox[0])
    ht = (bbox[3] - bbox[1])
    if (PatchScale > 0.85):
        ratio = (ht / width)
        if (width < ht):
            patchWidth = (PatchScale * width)
            patchHt = (patchWidth * ratio)
        else:
            patchHt = (PatchScale * ht)
            patchWidth = (patchHt / ratio)
        xmin = (bbox[0] + (random.uniform(0, 1) * (width - patchWidth)))
        ymin = (bbox[1] + (random.uniform(0, 1) * (ht - patchHt)))
        xmax = ((xmin + patchWidth) + 1)
        ymax = ((ymin + patchHt) + 1)
    else:
        xmin = max(1, min((bbox[0] + (np.random.normal((- 0.0142), 0.1158) * width)), (imgwidth - 3)))
        ymin = max(1, min((bbox[1] + (np.random.normal(0.0043, 0.068) * ht)), (imght - 3)))
        xmax = min(max((xmin + 2), (bbox[2] + (np.random.normal(0.0154, 0.1337) * width))), (imgwidth - 3))
        ymax = min(max((ymin + 2), (bbox[3] + (np.random.normal((- 0.0013), 0.0711) * ht))), (imght - 3))
    bbox[0] = xmin
    bbox[1] = ymin
    bbox[2] = xmax
    bbox[3] = ymax
    return bbox

def im_to_torch(img: np.ndarray):
    'Transform ndarray image to torch tensor.\n    Parameters\n    ----------\n    img: numpy.ndarray\n        An ndarray with shape: `(H, W, 3)`.\n    Returns\n    -------\n    jt.array\n        A tensor with shape: `(3, H, W)`.\n    '
    if img.max() > 1:
        img = np.float32(img / 255)
    img = to_torch(img)
    img = img.transpose(2, 0, 1)
    return img

def torch_to_im(img):
    'Transform torch tensor to ndarray image.\n    Parameters\n    ----------\n    img: jt.array\n        A tensor with shape: `(3, H, W)`.\n    Returns\n    -------\n    numpy.ndarray\n        An ndarray with shape: `(H, W, 3)`.\n    '
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))
    return img

def load_image(img_path):
    return im_to_torch(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))

def to_numpy(tensor):
    if isinstance(tensor, jt.Var):
        return tensor.numpy()
    elif (type(tensor).__module__ != 'numpy'):
        raise ValueError('Cannot convert {} to numpy array'.format(type(tensor)))
    return tensor

def to_torch(ndarray):
    if isinstance(ndarray, np.ndarray):
        return jt.array(ndarray)
    else:
        raise ValueError('Cannot convert {} to jittor Var'.format(type(ndarray)))
    return ndarray

def cv_cropBox(img, bbox, input_size):
    'Crop bbox from image by Affinetransform.\n    Parameters\n    ----------\n    img: jt.array\n        A tensor with shape: `(3, H, W)`.\n    bbox: list or tuple\n        [xmin, ymin, xmax, ymax].\n    input_size: tuple\n        Resulting image size, as (height, width).\n    Returns\n    -------\n    jt.array\n        A tensor with shape: `(3, height, width)`.\n    '
    (xmin, ymin, xmax, ymax) = bbox
    xmax -= 1
    ymax -= 1
    (resH, resW) = input_size
    lenH = max((ymax - ymin), (((xmax - xmin) * resH) / resW))
    lenW = ((lenH * resW) / resH)
    if (img.ndim == 2):
        img = img[np.newaxis, :, :]
    box_shape = [(ymax - ymin), (xmax - xmin)]
    pad_size = [((lenH - box_shape[0]) // 2), ((lenW - box_shape[1]) // 2)]
    (img[:, :ymin, :], img[:, :, :xmin]) = (0, 0)
    (img[:, (ymax + 1):, :], img[:, :, (xmax + 1):]) = (0, 0)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = np.array([(xmin - pad_size[1]), (ymin - pad_size[0])], np.float32)
    src[1, :] = np.array([(xmax + pad_size[1]), (ymax + pad_size[0])], np.float32)
    dst[0, :] = 0
    dst[1, :] = np.array([(resW - 1), (resH - 1)], np.float32)
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    dst_img = cv2.warpAffine(torch_to_im(img), trans, (resW, resH), flags=cv2.INTER_LINEAR)
    if (dst_img.ndim == 2):
        dst_img = dst_img[:, :, np.newaxis]
    return im_to_torch(jt.Var(dst_img))

def cv_cropBox_rot(img, bbox, input_size, rot):
    'Crop bbox from image by Affinetransform.\n    Parameters\n    ----------\n    img: jt.array\n        A tensor with shape: `(3, H, W)`.\n    bbox: list or tuple\n        [xmin, ymin, xmax, ymax].\n    input_size: tuple\n        Resulting image size, as (height, width).\n    Returns\n    -------\n    jt.array\n        A tensor with shape: `(3, height, width)`.\n    '
    (xmin, ymin, xmax, ymax) = bbox
    xmax -= 1
    ymax -= 1
    (resH, resW) = input_size
    rot_rad = ((np.pi * rot) / 180)
    if (img.ndim == 2):
        img = img[np.newaxis, :, :]
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    center = np.array([((xmax + xmin) / 2), ((ymax + ymin) / 2)])
    src_dir = get_dir([0, ((ymax - ymin) * (- 0.5))], rot_rad)
    dst_dir = np.array([0, ((resH - 1) * (- 0.5))], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = (center + src_dir)
    dst[0, :] = [((resW - 1) * 0.5), ((resH - 1) * 0.5)]
    dst[1, :] = (np.array([((resW - 1) * 0.5), ((resH - 1) * 0.5)]) + dst_dir)
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    dst_img = cv2.warpAffine(torch_to_im(img), trans, (resW, resH), flags=cv2.INTER_LINEAR)
    if (dst_img.ndim == 2):
        dst_img = dst_img[:, :, np.newaxis]
    return im_to_torch(jt.Var(dst_img))

def fix_cropBox(img, bbox, input_size):
    'Crop bbox from image by Affinetransform.\n    Parameters\n    ----------\n    img: jt.array\n        A tensor with shape: `(3, H, W)`.\n    bbox: list or tuple\n        [xmin, ymin, xmax, ymax].\n    input_size: tuple\n        Resulting image size, as (height, width).\n    Returns\n    -------\n    jt.array\n        A tensor with shape: `(3, height, width)`.\n    '
    (xmin, ymin, xmax, ymax) = bbox
    input_ratio = (input_size[0] / input_size[1])
    bbox_ratio = ((ymax - ymin) / (xmax - xmin))
    if (bbox_ratio > input_ratio):
        cx = ((xmax + xmin) / 2)
        h = (ymax - ymin)
        w = (h / input_ratio)
        xmin = (cx - (w / 2))
        xmax = (cx + (w / 2))
    elif (bbox_ratio < input_ratio):
        cy = ((ymax + ymin) / 2)
        w = (xmax - xmin)
        h = (w * input_ratio)
        ymin = (cy - (h / 2))
        ymax = (cy + (h / 2))
    bbox = [int(x) for x in [xmin, ymin, xmax, ymax]]
    return (cv_cropBox(img, bbox, input_size), bbox)

def fix_cropBox_rot(img, bbox, input_size, rot):
    'Crop bbox from image by Affinetransform.\n    Parameters\n    ----------\n    img: jt.array\n        A tensor with shape: `(3, H, W)`.\n    bbox: list or tuple\n        [xmin, ymin, xmax, ymax].\n    input_size: tuple\n        Resulting image size, as (height, width).\n    Returns\n    -------\n    jt.array\n        A tensor with shape: `(3, height, width)`.\n    '
    (xmin, ymin, xmax, ymax) = bbox
    input_ratio = (input_size[0] / input_size[1])
    bbox_ratio = ((ymax - ymin) / (xmax - xmin))
    if (bbox_ratio > input_ratio):
        cx = ((xmax + xmin) / 2)
        h = (ymax - ymin)
        w = (h / input_ratio)
        xmin = (cx - (w / 2))
        xmax = (cx + (w / 2))
    elif (bbox_ratio < input_ratio):
        cy = ((ymax + ymin) / 2)
        w = (xmax - xmin)
        h = (w * input_ratio)
        ymin = (cy - (h / 2))
        ymax = (cy + (h / 2))
    bbox = [int(x) for x in [xmin, ymin, xmax, ymax]]
    return (cv_cropBox_rot(img, bbox, input_size, rot), bbox)

def get_3rd_point(a, b):
    'Return vector c that perpendicular to (a - b).'
    direct = (a - b)
    return (b + np.array([(- direct[1]), direct[0]], dtype=np.float32))

def get_dir(src_point, rot_rad):
    'Rotate the point by `rot_rad` degree.'
    (sn, cs) = (np.sin(rot_rad), np.cos(rot_rad))
    src_result = [0, 0]
    src_result[0] = ((src_point[0] * cs) - (src_point[1] * sn))
    src_result[1] = ((src_point[0] * sn) + (src_point[1] * cs))
    return src_result

def cv_cropBoxInverse(inp, bbox, img_size, output_size):
    'Paste the cropped bbox to the original image.\n    Parameters\n    ----------\n    inp: jt.array\n        A tensor with shape: `(3, height, width)`.\n    bbox: list or tuple\n        [xmin, ymin, xmax, ymax].\n    img_size: tuple\n        Original image size, as (img_H, img_W).\n    output_size: tuple\n        Cropped input size, as (height, width).\n    Returns\n    -------\n    jt.array\n        A tensor with shape: `(3, img_H, img_W)`.\n    '
    (xmin, ymin, xmax, ymax) = bbox
    xmax -= 1
    ymax -= 1
    (resH, resW) = output_size
    (imgH, imgW) = img_size
    lenH = max((ymax - ymin), (((xmax - xmin) * resH) / resW))
    lenW = ((lenH * resW) / resH)
    if (inp.ndim == 2):
        inp = inp[np.newaxis, :, :]
    box_shape = [(ymax - ymin), (xmax - xmin)]
    pad_size = [((lenH - box_shape[0]) // 2), ((lenW - box_shape[1]) // 2)]
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = 0
    src[1, :] = np.array([(resW - 1), (resH - 1)], np.float32)
    dst[0, :] = np.array([(xmin - pad_size[1]), (ymin - pad_size[0])], np.float32)
    dst[1, :] = np.array([(xmax + pad_size[1]), (ymax + pad_size[0])], np.float32)
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    dst_img = cv2.warpAffine(torch_to_im(inp), trans, (imgW, imgH), flags=cv2.INTER_LINEAR)
    if ((dst_img.ndim == 3) and (dst_img.shape[2] == 1)):
        dst_img = dst_img[:, :, 0]
        return dst_img
    elif (dst_img.ndim == 2):
        return dst_img
    else:
        return im_to_torch(jt.Var(dst_img))

def cv_rotate(img, rot, input_size):
    'Rotate image by Affinetransform.\n    Parameters\n    ----------\n    img: jt.array\n        A tensor with shape: `(3, H, W)`.\n    rot: int\n        Rotation degree.\n    input_size: tuple\n        Resulting image size, as (height, width).\n    Returns\n    -------\n    jt.array\n        A tensor with shape: `(3, height, width)`.\n    '
    (resH, resW) = input_size
    center = (np.array(((resW - 1), (resH - 1))) / 2)
    rot_rad = ((np.pi * rot) / 180)
    src_dir = get_dir([0, ((resH - 1) * (- 0.5))], rot_rad)
    dst_dir = np.array([0, ((resH - 1) * (- 0.5))], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = (center + src_dir)
    dst[0, :] = [((resW - 1) * 0.5), ((resH - 1) * 0.5)]
    dst[1, :] = (np.array([((resW - 1) * 0.5), ((resH - 1) * 0.5)]) + dst_dir)
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    dst_img = cv2.warpAffine(torch_to_im(img), trans, (resW, resH), flags=cv2.INTER_LINEAR)
    if (dst_img.ndim == 2):
        dst_img = dst_img[:, :, np.newaxis]
    return im_to_torch(jt.Var(dst_img))

def count_visible(bbox, joints_3d):
    'Count number of visible joints given bound box.'
    vis = np.logical_and.reduce(((joints_3d[:, 0, 0] > 0), (joints_3d[:, 0, 0] > bbox[0]), (joints_3d[:, 0, 0] < bbox[2]), (joints_3d[:, 1, 0] > 0), (joints_3d[:, 1, 0] > bbox[1]), (joints_3d[:, 1, 0] < bbox[3]), (joints_3d[:, 0, 1] > 0), (joints_3d[:, 1, 1] > 0)))
    return (np.sum(vis), vis)

def drawGaussian(img, pt, sigma):
    'Draw 2d gaussian on input image.\n    Parameters\n    ----------\n    img: jt.array\n        A tensor with shape: `(3, H, W)`.\n    pt: list or tuple\n        A point: (x, y).\n    sigma: int\n        Sigma of gaussian distribution.\n    Returns\n    -------\n    jt.array\n        A tensor with shape: `(3, H, W)`.\n    '
    img = to_numpy(img)
    tmpSize = (3 * sigma)
    ul = [int((pt[0] - tmpSize)), int((pt[1] - tmpSize))]
    br = [int(((pt[0] + tmpSize) + 1)), int(((pt[1] + tmpSize) + 1))]
    if ((ul[0] >= img.shape[1]) or (ul[1] >= img.shape[0]) or (br[0] < 0) or (br[1] < 0)):
        return to_torch(img)
    size = ((2 * tmpSize) + 1)
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = (size // 2)
    g = np.exp(((- (((x - x0) ** 2) + ((y - y0) ** 2))) / (2 * (sigma ** 2))))
    g_x = (max(0, (- ul[0])), (min(br[0], img.shape[1]) - ul[0]))
    g_y = (max(0, (- ul[1])), (min(br[1], img.shape[0]) - ul[1]))
    img_x = (max(0, ul[0]), min(br[0], img.shape[1]))
    img_y = (max(0, ul[1]), min(br[1], img.shape[0]))
    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return to_torch(img)

def flip(x):
    assert ((x.ndim == 3) or (x.ndim == 4))
    dim = (x.ndim - 1)
    return x.flip(dims=(dim,))

def flip_heatmap(heatmap, joint_pairs, shift=False):
    'Flip pose heatmap according to joint pairs.\n    Parameters\n    ----------\n    heatmap : numpy.ndarray\n        Heatmap of joints.\n    joint_pairs : list\n        List of joint pairs.\n    shift : bool\n        Whether to shift the output.\n    Returns\n    -------\n    numpy.ndarray\n        Flipped heatmap.\n    '
    assert ((heatmap.ndim == 3) or (heatmap.ndim == 4))
    out = flip(heatmap)
    for pair in joint_pairs:
        (dim0, dim1) = pair
        idx = jt.array((dim0, dim1)).long()
        inv_idx = jt.array((dim1, dim0)).long()
        if (out.ndim == 4):
            out[:, idx] = out[:, inv_idx]
        else:
            out[idx] = out[inv_idx]
    if shift:
        if (out.ndim == 3):
            out[:, :, 1:] = out[:, :, 0:(- 1)]
        else:
            out[:, :, :, 1:] = out[:, :, :, 0:(- 1)]
    return out

def flip_joints_3d(joints_3d, width, joint_pairs):
    'Flip 3d joints.\n    Parameters\n    ----------\n    joints_3d : numpy.ndarray\n        Joints in shape (num_joints, 3, 2)\n    width : int\n        Image width.\n    joint_pairs : list\n        List of joint pairs.\n    Returns\n    -------\n    numpy.ndarray\n        Flipped 3d joints with shape (num_joints, 3, 2)\n    '
    joints = joints_3d.copy()
    joints[:, 0, 0] = ((width - joints[:, 0, 0]) - 1)
    for pair in joint_pairs:
        (joints[pair[0], :, 0], joints[pair[1], :, 0]) = (joints[pair[1], :, 0], joints[pair[0], :, 0].copy())
        (joints[pair[0], :, 1], joints[pair[1], :, 1]) = (joints[pair[1], :, 1], joints[pair[0], :, 1].copy())
    joints[:, :, 0] *= joints[:, :, 1]
    return joints

def heatmap_to_coord_simple(hms, bbox, hms_flip=None, **kwargs):
    if (hms_flip is not None):
        hms = ((hms + hms_flip) / 2)
    (coords, maxvals) = get_max_pred(hms)
    hm_h = hms.shape[1]
    hm_w = hms.shape[2]
    for p in range(coords.shape[0]):
        hm = hms[p]
        px = int(round(float(coords[p][0])))
        py = int(round(float(coords[p][1])))
        if ((1 < px < (hm_w - 1)) and (1 < py < (hm_h - 1))):
            diff = np.array(((hm[py][(px + 1)] - hm[py][(px - 1)]), (hm[(py + 1)][px] - hm[(py - 1)][px])))
            coords[p] += (np.sign(diff) * 0.25)
    preds = np.zeros_like(coords)
    (xmin, ymin, xmax, ymax) = bbox
    w = (xmax - xmin)
    h = (ymax - ymin)
    center = np.array([(xmin + (w * 0.5)), (ymin + (h * 0.5))])
    scale = np.array([w, h])
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center, scale, [hm_w, hm_h])
    return (preds, maxvals)

def heatmap_to_coord_simple_regress(preds, bbox, hm_shape, norm_type, hms_flip=None):

    def integral_op(hm_1d):
        if (hm_1d.device.index is not None):
            # hm_1d = (hm_1d * torch.cuda.comm.broadcast(jt.arange(hm_1d.shape[(- 1)]).astype(jt.float32), devices=[hm_1d.device.index])[0])
            hm_1d = hm_1d * jt.arange(hm_1d.shape[(- 1)]).astype(jt.float32)  # tycoer
        else:
            hm_1d = (hm_1d * jt.arange(hm_1d.shape[(- 1)]).astype(jt.float32))
        return hm_1d
    if (preds.ndim == 3):
        preds = preds.unsqueeze(0)
    (hm_height, hm_width) = hm_shape
    num_joints = preds.shape[1]
    (pred_jts, pred_scores) = _integral_tensor(preds, num_joints, False, hm_width, hm_height, 1, integral_op, norm_type)
    pred_jts = pred_jts.reshape((pred_jts.shape[0], num_joints, 2))
    if (hms_flip is not None):
        if (hms_flip.ndim == 3):
            hms_flip = hms_flip.unsqueeze(0)
        (pred_jts_flip, pred_scores_flip) = _integral_tensor(hms_flip, num_joints, False, hm_width, hm_height, 1, integral_op, norm_type)
        pred_jts_flip = pred_jts_flip.reshape((pred_jts_flip.shape[0], num_joints, 2))
        pred_jts = ((pred_jts + pred_jts_flip) / 2)
        pred_scores = ((pred_scores + pred_scores_flip) / 2)
    ndims = pred_jts.ndim
    assert (ndims in [2, 3]), 'Dimensions of input heatmap should be 3 or 4'
    if (ndims == 2):
        pred_jts = pred_jts.unsqueeze(0)
        pred_scores = pred_scores.unsqueeze(0)
    coords = pred_jts.numpy()
    coords = coords.astype(np.float32)
    pred_scores = pred_scores.numpy()
    pred_scores = pred_scores.astype(np.float32)
    coords[:, :, 0] = ((coords[:, :, 0] + 0.5) * hm_width)
    coords[:, :, 1] = ((coords[:, :, 1] + 0.5) * hm_height)
    preds = np.zeros_like(coords)
    (xmin, ymin, xmax, ymax) = bbox
    w = (xmax - xmin)
    h = (ymax - ymin)
    center = np.array([(xmin + (w * 0.5)), (ymin + (h * 0.5))])
    scale = np.array([w, h])
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            preds[i, j, 0:2] = transform_preds(coords[i, j, 0:2], center, scale, [hm_width, hm_height])
    if (preds.shape[0] == 1):
        preds = preds[0]
        pred_scores = pred_scores[0]
    return (preds, pred_scores)

def _integral_tensor(preds, num_joints, output_3d, hm_width, hm_height, hm_depth, integral_operation, norm_type='softmax'):
    preds = preds.reshape((preds.shape[0], num_joints, (- 1)))
    preds = norm_heatmap(norm_type, preds)
    if (norm_type == 'sigmoid'):
        (maxvals, _) = jt.max(preds, dim=2, keepdims=True)
    else:
        maxvals = jt.ones((*preds.shape[:2], 1), dtype=jt.float)
    heatmaps = (preds / preds.sum(dim=2, keepdims=True))
    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, hm_depth, hm_height, hm_width))
    hm_x = heatmaps.sum(dim=(2, 3))
    hm_y = heatmaps.sum(dim=(2, 4))
    hm_z = heatmaps.sum(dim=(3, 4))
    hm_x = integral_operation(hm_x)
    hm_y = integral_operation(hm_y)
    hm_z = integral_operation(hm_z)
    coord_x = hm_x.sum(dim=2, keepdims=True)
    coord_y = hm_y.sum(dim=2, keepdims=True)
    coord_z = hm_z.sum(dim=2, keepdims=True)
    coord_x = ((coord_x / float(hm_width)) - 0.5)
    coord_y = ((coord_y / float(hm_height)) - 0.5)
    if output_3d:
        coord_z = ((coord_z / float(hm_depth)) - 0.5)
        pred_jts = jt.contrib.concat((coord_x, coord_y, coord_z), dim=2)
        pred_jts = pred_jts.reshape((pred_jts.shape[0], (num_joints * 3)))
    else:
        pred_jts = jt.contrib.concat((coord_x, coord_y), dim=2)
        pred_jts = pred_jts.reshape((pred_jts.shape[0], (num_joints * 2)))
    return (pred_jts, maxvals.float())

def norm_heatmap(norm_type, heatmap):
    shape = heatmap.shape
    if (norm_type == 'softmax'):
        heatmap = heatmap.reshape((*shape[:2], (- 1)))
        heatmap = nn.softmax(heatmap, dim=2)
        return heatmap.reshape(*shape)
    elif (norm_type == 'sigmoid'):
        return heatmap.sigmoid()
    elif (norm_type == 'divide_sum'):
        heatmap = heatmap.reshape((*shape[:2], (- 1)))
        heatmap = (heatmap / heatmap.sum(dim=2, keepdims=True))
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords

def get_max_pred(heatmaps):
    num_joints = heatmaps.shape[0]
    width = heatmaps.shape[2]
    heatmaps_reshaped = heatmaps.reshape((num_joints, (- 1)))
    idx = np.argmax(heatmaps_reshaped, 1)
    maxvals = np.max(heatmaps_reshaped, 1)
    maxvals = maxvals.reshape((num_joints, 1))
    idx = idx.reshape((num_joints, 1))
    preds = np.tile(idx, (1, 2)).astype(np.float32)
    preds[:, 0] = (preds[:, 0] % width)
    preds[:, 1] = np.floor((preds[:, 1] / width))
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 2))
    pred_mask = pred_mask.astype(np.float32)
    preds *= pred_mask
    return (preds, maxvals)

def get_max_pred_batch(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, (- 1)))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.max(heatmaps_reshaped, 2)
    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0] % width)
    preds[:, :, 1] = np.floor((preds[:, :, 1] / width))
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)
    preds *= pred_mask
    return (preds, maxvals)

def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if ((not isinstance(scale, np.ndarray)) and (not isinstance(scale, list))):
        scale = np.array([scale, scale])
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    rot_rad = ((np.pi * rot) / 180)
    src_dir = get_dir([0, (src_w * (- 0.5))], rot_rad)
    dst_dir = np.array([0, (dst_w * (- 0.5))], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = (center + (scale_tmp * shift))
    src[1, :] = ((center + src_dir) + (scale_tmp * shift))
    dst[0, :] = [(dst_w * 0.5), (dst_h * 0.5)]
    dst[1, :] = (np.array([(dst_w * 0.5), (dst_h * 0.5)]) + dst_dir)
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def flip_thetas(thetas, theta_pairs):
    'Flip thetas.\n\n    Parameters\n    ----------\n    thetas : numpy.ndarray\n        Joints in shape (num_thetas, 3)\n    theta_pairs : list\n        List of theta pairs.\n\n    Returns\n    -------\n    numpy.ndarray\n        Flipped thetas with shape (num_thetas, 3)\n\n    '
    thetas_flip = thetas.copy()
    thetas_flip[:, 1] = ((- 1) * thetas_flip[:, 1])
    thetas_flip[:, 2] = ((- 1) * thetas_flip[:, 2])
    for pair in theta_pairs:
        (thetas_flip[pair[0], :], thetas_flip[pair[1], :]) = (thetas_flip[pair[1], :], thetas_flip[pair[0], :].copy())
    return thetas_flip

def flip_xyz_joints_3d(joints_3d, joint_pairs):
    'Flip 3d xyz joints.\n\n    Parameters\n    ----------\n    joints_3d : numpy.ndarray\n        Joints in shape (num_joints, 3)\n    joint_pairs : list\n        List of joint pairs.\n\n    Returns\n    -------\n    numpy.ndarray\n        Flipped 3d joints with shape (num_joints, 3)\n\n    '
    assert (joints_3d.ndim in (2, 3))
    joints = joints_3d.copy()
    joints[:, 0] = ((- 1) * joints[:, 0])
    for pair in joint_pairs:
        (joints[pair[0], :], joints[pair[1], :]) = (joints[pair[1], :], joints[pair[0], :].copy())
    return joints

def batch_rodrigues_numpy(rot_vecs, epsilon=1e-08):
    ' Calculates the rotation matrices for a batch of rotation vectors\n        Parameters\n        ----------\n        rot_vecs: numpy.ndarray Nx3\n            array of N axis-angle vectors\n        Returns\n        -------\n        R: numpy.ndarray Nx3x3\n            The rotation matrices for the given axis-angle parameters\n    '
    batch_size = rot_vecs.shape[0]
    angle = np.linalg.norm((rot_vecs + 1e-08), axis=1, keepdims=True)
    rot_dir = (rot_vecs / angle)
    cos = np.cos(angle)[:, None, :]
    sin = np.sin(angle)[:, None, :]
    (rx, ry, rz) = np.split(rot_dir, 3, axis=1)
    K = np.zeros((batch_size, 3, 3))
    zeros = np.zeros((batch_size, 1))
    K = np.concatenate([zeros, (- rz), ry, rz, zeros, (- rx), (- ry), rx, zeros], axis=1).reshape((batch_size, 3, 3))
    ident = np.eye(3)[None, :, :]
    rot_mat = ((ident + (sin * K)) + ((1 - cos) * np.einsum('bij,bjk->bik', K, K)))
    return rot_mat

def rotmat_to_quat_numpy(rotmat):
    'Convert quaternion coefficients to rotation matrix.\n    Args:\n        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]\n    Returns:\n        quat: size = [B, 4] 4 <===>(w, x, y, z)\n    '
    trace = np.einsum('bii->b', rotmat)
    m32 = rotmat[:, 2, 1]
    m23 = rotmat[:, 1, 2]
    m13 = rotmat[:, 0, 2]
    m31 = rotmat[:, 2, 0]
    m21 = rotmat[:, 1, 0]
    m12 = rotmat[:, 0, 1]
    trace = (trace + 1)
    w = (np.sqrt(trace.clip(min=1e-08)) / 2)
    x = ((m32 - m23) / (4 * w))
    y = ((m13 - m31) / (4 * w))
    z = ((m21 - m12) / (4 * w))
    return np.stack([w, x, y, z], axis=1)

def flip_twist(twist_phi, twist_weight, twist_pairs):
    twist_flip = np.zeros_like(twist_phi)
    weight_flip = twist_weight.copy()
    twist_flip[:, 0] = twist_phi[:, 0].copy()
    twist_flip[:, 1] = ((- 1) * twist_phi[:, 1].copy())
    for pair in twist_pairs:
        idx0 = (pair[0] - 1)
        idx1 = (pair[1] - 1)
        (twist_flip[idx0, :], twist_flip[idx1, :]) = (twist_flip[idx1, :], twist_flip[idx0, :].copy())
        (weight_flip[idx0, :], weight_flip[idx1, :]) = (weight_flip[idx1, :], weight_flip[idx0, :].copy())
    return (twist_flip, weight_flip)

def get_intrinsic_metrix(f, c, inv=False):
    intrinsic_metrix = np.zeros((3, 3)).astype(np.float32)
    if inv:
        intrinsic_metrix[(0, 0)] = (1.0 / f[0])
        intrinsic_metrix[(0, 2)] = ((- c[0]) / f[0])
        intrinsic_metrix[(1, 1)] = (1.0 / f[1])
        intrinsic_metrix[(1, 2)] = ((- c[1]) / f[1])
        intrinsic_metrix[(2, 2)] = 1
    else:
        intrinsic_metrix[(0, 0)] = f[0]
        intrinsic_metrix[(0, 2)] = c[0]
        intrinsic_metrix[(1, 1)] = f[1]
        intrinsic_metrix[(1, 2)] = c[1]
        intrinsic_metrix[(2, 2)] = 1
    return intrinsic_metrix

def get_func_heatmap_to_coord(cfg):
    if (cfg.DATA_PRESET.TYPE == 'simple'):
        if (cfg.LOSS.TYPE == 'MSELoss'):
            return heatmap_to_coord_simple
        elif (cfg.LOSS.TYPE == 'L1JointRegression'):
            return heatmap_to_coord_simple_regress
        elif (cfg.LOSS.TYPE == 'Combined'):
            return [heatmap_to_coord_simple, heatmap_to_coord_simple_regress]
    else:
        raise NotImplementedError
