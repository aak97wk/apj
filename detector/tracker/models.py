
import jittor as jt
from jittor import init
import os
from collections import defaultdict, OrderedDict
from jittor import nn
from tracker.utils.parse_config import *
from tracker.utils.utils import *
import time
import math
import numpy as np
batch_norm = nn.BatchNorm2d

def create_modules(module_defs):
    '\n    Constructs module list of layer blocks from module configuration in module_defs\n    '
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    yolo_layer_count = 0
    for (i, module_def) in enumerate(module_defs):
        modules = nn.Sequential()
        if (module_def['type'] == 'convolutional'):
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (((kernel_size - 1) // 2) if int(module_def['pad']) else 0)
            modules.add_module(('conv_%d' % i), nn.Conv(output_filters[(- 1)], filters, kernel_size, stride=int(module_def['stride']), padding=pad, bias=(not bn)))
            if bn:
                modules.add_module(('batch_norm_%d' % i), batch_norm(filters))
            if (module_def['activation'] == 'leaky'):
                modules.add_module(('leaky_%d' % i), nn.LeakyReLU(scale=0.1))
        elif (module_def['type'] == 'maxpool'):
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if ((kernel_size == 2) and (stride == 1)):
                modules.add_module(('_debug_padding_%d' % i), jt.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.Pool(kernel_size, stride=stride, padding=int(((kernel_size - 1) // 2)), op='maximum')
            modules.add_module(('maxpool_%d' % i), maxpool)
        elif (module_def['type'] == 'upsample'):
            upsample = Upsample(scale_factor=int(module_def['stride']))
            modules.add_module(('upsample_%d' % i), upsample)
        elif (module_def['type'] == 'route'):
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[((i + 1) if (i > 0) else i)] for i in layers])
            modules.add_module(('route_%d' % i), EmptyLayer())
        elif (module_def['type'] == 'shortcut'):
            filters = output_filters[int(module_def['from'])]
            modules.add_module(('shortcut_%d' % i), EmptyLayer())
        elif (module_def['type'] == 'yolo'):
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            anchors = [float(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[(i + 1)]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            nC = int(module_def['classes'])
            img_size = (int(hyperparams['width']), int(hyperparams['height']))
            yolo_layer = YOLOLayer(anchors, nC, hyperparams['nID'], img_size, yolo_layer_count, cfg=hyperparams['cfg'])
            modules.add_module(('yolo_%d' % i), yolo_layer)
            yolo_layer_count += 1
        module_list.append(modules)
        output_filters.append(filters)
    return (hyperparams, module_list)

class EmptyLayer(nn.Module):
    "Placeholder for 'route' and 'shortcut' layers"

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def execute(self, x):
        return x

class Upsample(nn.Module):

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def execute(self, x):
        return nn.interpolate(x, scale_factor=self.scale_factor, mode='nearest')

class YOLOLayer(nn.Module):

    def __init__(self, anchors, nC, nID, img_size, yolo_layer, cfg):
        super(YOLOLayer, self).__init__()
        self.layer = yolo_layer
        nA = len(anchors)
        self.anchors = jt.float32(anchors)
        self.nA = nA
        self.nC = nC
        self.nID = nID
        self.img_size = 0
        self.emb_dim = 512
        # self.SmoothL1Loss =
        raise RuntimeError('original source: <nn.SmoothL1Loss()>, SmoothL1Loss is not supported in Jittor yet. We will appreciate it if you provide an implementation of SmoothL1Loss and make pull request at https://github.com/Jittor/jittor.')
        self.SoftmaxLoss = nn.CrossEntropyLoss(ignore_index=(- 1))
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=(- 1))
        self.s_c = jt.array(((- 4.15) * jt.ones(1)))
        self.s_r = jt.array(((- 4.85) * jt.ones(1)))
        self.s_id = jt.array(((- 2.3) * jt.ones(1)))
        self.emb_scale = (math.sqrt(2) * math.log((self.nID - 1)))

    def execute(self, p_cat, img_size, targets=None, classifier=None, test_emb=False):
        (p, p_emb) = (p_cat[:, :24, ...], p_cat[:, 24:, ...])
        (nB, nGh, nGw) = (p.shape[0], p.shape[(- 2)], p.shape[(- 1)])
        if (self.img_size != img_size):
            create_grids(self, img_size, nGh, nGw)
            self.grid_xy = self.grid_xy.type_as(p)
            self.anchor_wh = self.anchor_wh.type_as(p)
        p = p.view(nB, self.nA, (self.nC + 5), nGh, nGw).permute((0, 1, 3, 4, 2))
        p_emb = p_emb.permute((0, 2, 3, 1))
        p_box = p[..., :4]
        p_conf = p[..., 4:6].permute((0, 4, 1, 2, 3))
        if (targets is not None):
            if test_emb:
                (tconf, tbox, tids) = build_targets_max(targets, self.anchor_vec, self.nA, self.nC, nGh, nGw)
            else:
                (tconf, tbox, tids) = build_targets_thres(targets, self.anchor_vec, self.nA, self.nC, nGh, nGw)
            # (tconf, tbox, tids) = (tconf, tbox, tids.cuda())
            mask = (tconf > 0)
            nT = sum([len(x) for x in targets])
            nM = mask.sum().float()
            nP = jt.ones_like(mask).sum().float()
            if (nM > 0):
                lbox = jt.nn.smooth_l1_loss(p_box[mask], tbox[mask])
            else:
                FT = jt.float32
                (lbox, lconf) = (FT([0]), FT([0]))
            lconf = self.SoftmaxLoss(p_conf, tconf)
            lid = jt.Var(1).fill_(0).squeeze(-1)
            (emb_mask, _) = mask.max(dim=1)
            (tids, _) = tids.max(dim=1)
            tids = tids[emb_mask]
            embedding = p_emb[emb_mask]
            embedding = (self.emb_scale * F.normalize(embedding))
            nI = emb_mask.sum().float()
            if test_emb:
                if ((np.prod(embedding.shape) == 0) or (np.prod(tids.shape) == 0)):
                    return jt.zeros((0, (self.emb_dim + 1)))
                emb_and_gt = jt.contrib.concat([embedding, tids.float()], dim=1)
                return emb_and_gt
            if (len(embedding) > 1):
                logits = classifier(embedding)
                lid = self.IDLoss(logits, tids.squeeze(-1))
            loss = ((((jt.exp((- self.s_r)) * lbox) + (jt.exp((- self.s_c)) * lconf)) + (jt.exp((- self.s_id)) * lid)) + ((self.s_r + self.s_c) + self.s_id))
            loss *= 0.5
            return (loss, loss.item(), lbox.item(), lconf.item(), lid.item(), nT)
        else:
            p_conf = nn.softmax(p_conf, dim=1)[:, 1, ...].unsqueeze((- 1))
            p_emb = p_emb.unsqueeze(1).repeat(1, self.nA, 1, 1, 1)
            p_cls = jt.zeros((nB, self.nA, nGh, nGw, 1)).astype(p)
            p = jt.contrib.concat([p_box, p_conf, p_cls, p_emb], dim=(- 1))
            p[..., :4] = decode_delta_map(p[..., :4], self.anchor_vec.astype(p))
            p[..., :4] *= self.stride
            return p.view((nB, (- 1), p.shape[(- 1)]))

class Darknet(nn.Module):
    'YOLOv3 object detection model'

    def __init__(self, cfg_path, img_size=(1088, 608), nID=1591, test_emb=False):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_cfg(cfg_path)
        self.module_defs[0]['cfg'] = cfg_path
        self.module_defs[0]['nID'] = nID
        (self.hyperparams, self.module_list) = create_modules(self.module_defs)
        self.img_size = img_size
        self.loss_names = ['loss', 'box', 'conf', 'id', 'nT']
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        self.emb_dim = 512
        self.classifier = nn.Linear(self.emb_dim, nID)
        self.test_emb = test_emb

    def execute(self, x, targets=None, targets_len=None):
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        is_training = ((targets is not None) and (not self.test_emb))
        layer_outputs = []
        output = []
        for (i, (module_def, module)) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def['type']
            if (mtype in ['convolutional', 'upsample', 'maxpool']):
                x = module(x)
            elif (mtype == 'route'):
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if (len(layer_i) == 1):
                    x = layer_outputs[layer_i[0]]
                else:
                    x = jt.contrib.concat([layer_outputs[i] for i in layer_i], dim=1)
            elif (mtype == 'shortcut'):
                layer_i = int(module_def['from'])
                x = (layer_outputs[(- 1)] + layer_outputs[layer_i])
            elif (mtype == 'yolo'):
                if is_training:
                    targets = [targets[i][:int(l)] for (i, l) in enumerate(targets_len)]
                    (x, *losses) = module[0](x, self.img_size, targets, self.classifier)
                    for (name, loss) in zip(self.loss_names, losses):
                        self.losses[name] += loss
                elif self.test_emb:
                    targets = [targets[i][:int(l)] for (i, l) in enumerate(targets_len)]
                    x = module[0](x, self.img_size, targets, self.classifier, self.test_emb)
                else:
                    x = module[0](x, self.img_size)
                output.append(x)
            layer_outputs.append(x)
        if is_training:
            self.losses['nT'] /= 3
            output = [o.squeeze(-1) for o in output]
            return (sum(output), jt.Var(list(self.losses.values())))
        elif self.test_emb:
            return jt.contrib.concat(output, dim=0)
        return jt.contrib.concat(output, dim=1)

def create_grids(self, img_size, nGh, nGw):
    self.stride = (img_size[0] / nGw)
    assert (self.stride == (img_size[1] / nGh))
    grid_x = jt.arange(nGw).repeat((nGh, 1)).view((1, 1, nGh, nGw)).float()
    grid_y = jt.arange(nGh).repeat((nGw, 1)).transpose(0, 1).view((1, 1, nGh, nGw)).float()
    self.grid_xy = jt.stack((grid_x, grid_y), dim=4)
    self.anchor_vec = (self.anchors / self.stride)
    self.anchor_wh = self.anchor_vec.view((1, self.nA, 1, 1, 2))

def load_darknet_weights(self, weights, cutoff=(- 1)):
    weights_file = weights.split(os.sep)[(- 1)]
    if (not os.path.isfile(weights)):
        try:
            os.system(((('wget https://pjreddie.com/media/files/' + weights_file) + ' -O ') + weights))
        except IOError:
            print((weights + ' not found'))
    if (weights_file == 'darknet53.conv.74'):
        cutoff = 75
    elif (weights_file == 'yolov3-tiny.conv.15'):
        cutoff = 15
    fp = open(weights, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)
    self.header_info = header
    self.seen = header[3]
    weights = np.fromfile(fp, dtype=np.float32)
    fp.close()
    ptr = 0
    for (i, (module_def, module)) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if (module_def['type'] == 'convolutional'):
            conv_layer = module[0]
            if module_def['batch_normalize']:
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()
                bn_b = jt.array(weights[ptr:(ptr + num_b)]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                bn_w = jt.array(weights[ptr:(ptr + num_b)]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                bn_rm = jt.array(weights[ptr:(ptr + num_b)]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                bn_rv = jt.array(weights[ptr:(ptr + num_b)]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                num_b = conv_layer.bias.numel()
                conv_b = jt.array(weights[ptr:(ptr + num_b)]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            num_w = conv_layer.weight.numel()
            conv_w = jt.array(weights[ptr:(ptr + num_w)]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w
'\n    @:param path    - path of the new weights file\n    @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)\n'

def save_weights(self, path, cutoff=(- 1)):
    fp = open(path, 'wb')
    self.header_info[3] = self.seen
    self.header_info.tofile(fp)
    for (i, (module_def, module)) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if (module_def['type'] == 'convolutional'):
            conv_layer = module[0]
            if module_def['batch_normalize']:
                bn_layer = module[1]
                bn_layer.bias.data.numpy().tofile(fp)
                bn_layer.weight.data.numpy().tofile(fp)
                bn_layer.running_mean.numpy().tofile(fp)
                bn_layer.running_var.numpy().tofile(fp)
            else:
                conv_layer.bias.numpy().tofile(fp)
            conv_layer.weight.numpy().tofile(fp)
    fp.close()
