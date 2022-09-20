
import jittor as jt
from jittor import init
from jittor import nn
'API of yolo detector'
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from abc import ABC, abstractmethod
import platform
import numpy as np
from yolo.preprocess import prep_image, prep_frame
from yolo.darknet import Darknet
from yolo.util import unique
from yolo.bbox import bbox_iou
from detector.apis import BaseDetector
if (platform.system() != 'Windows'):
    from detector.nms import nms_wrapper

class YOLODetector(BaseDetector):

    def __init__(self, cfg, opt=None):
        super(YOLODetector, self).__init__()
        self.detector_cfg = cfg
        self.detector_opt = opt
        self.model_cfg = cfg.get('CONFIG', 'detector/yolo/cfg/yolov3-spp.cfg')
        self.model_weights = cfg.get('WEIGHTS', 'detector/yolo/data/yolov3-spp.weights')
        self.inp_dim = cfg.get('INP_DIM', 608)
        self.nms_thres = cfg.get('NMS_THRES', 0.6)
        self.confidence = (0.3 if (False if (not hasattr(opt, 'tracking')) else opt.tracking) else cfg.get('CONFIDENCE', 0.05))
        self.num_classes = cfg.get('NUM_CLASSES', 80)
        self.model = None

    def load_model(self):
        args = self.detector_opt
        print('Loading YOLO model..')
        self.model = Darknet(self.model_cfg)
        self.model.load_weights(self.model_weights)
        self.model.net_info['height'] = self.inp_dim
        # if args:
            # if (len(args.gpus) > 1):
            #     self.model = torch.nn.DataParallel(self.model, device_ids=args.gpus).to(args.device)
            # else:
            #     self.model.to(args.device)
        # else:
            # self.model
        self.model.eval()

    def image_preprocess(self, img_source):
        '\n        Pre-process the img before fed to the object detection network\n        Input: image name(str) or raw image data(ndarray or jt.array,channel GBR)\n        Output: pre-processed image data(torch.FloatTensor,(1,3,h,w))\n        '
        if isinstance(img_source, str):
            (img, orig_img, im_dim_list) = prep_image(img_source, self.inp_dim)
        elif (isinstance(img_source, jt.Var) or isinstance(img_source, np.ndarray)):
            (img, orig_img, im_dim_list) = prep_frame(img_source, self.inp_dim)
        else:
            raise IOError('Unknown image source type: {}'.format(type(img_source)))
        return img

    def images_detection(self, imgs, orig_dim_list):
        '\n        Feed the img data into object detection network and \n        collect bbox w.r.t original image size\n        Input: imgs(torch.FloatTensor,(b,3,h,w)): pre-processed mini-batch image input\n               orig_dim_list(torch.FloatTensor, (b,(w,h,w,h))): original mini-batch image size\n        Output: dets(torch.cuda.FloatTensor,(n,(batch_idx,x1,y1,x2,y2,c,s,idx of cls))): human detection results\n        '
        args = self.detector_opt
        _CUDA = True
        if args:
            if (args.gpus[0] < 0):
                _CUDA = False
        if (not self.model):
            self.load_model()
        with jt.no_grad():
            # imgs = (imgs.to(args.device) if args else imgs.cuda())
            prediction = self.model(imgs, args=args)
            dets = self.dynamic_write_results(prediction, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thres)
            if (isinstance(dets, int) or (dets.shape[0] == 0)):
                return 0
            orig_dim_list = orig_dim_list[dets[:, 0].long()]
            scaling_factor = jt.min((self.inp_dim / orig_dim_list), 1)[0].view((- 1), 1)
            dets[:, [1, 3]] -= ((self.inp_dim - (scaling_factor * orig_dim_list[:, 0].view(((- 1), 1)))) / 2)
            dets[:, [2, 4]] -= ((self.inp_dim - (scaling_factor * orig_dim_list[:, 1].view(((- 1), 1)))) / 2)
            dets[:, 1:5] /= scaling_factor
            for i in range(dets.shape[0]):
                dets[(i, [1, 3])] = jt.clamp(dets[(i, [1, 3])], min_v=0.0, max_v=orig_dim_list[(i, 0)])
                dets[(i, [2, 4])] = jt.clamp(dets[(i, [2, 4])], min_v=0.0, max_v=orig_dim_list[(i, 1)])
            return dets

    def dynamic_write_results(self, prediction, confidence, num_classes, nms=True, nms_conf=0.4):
        prediction_bak = prediction.clone()
        dets = self.write_results(prediction.clone(), confidence, num_classes, nms, nms_conf)
        if isinstance(dets, int):
            return dets
        if (dets.shape[0] > 100):
            nms_conf -= 0.05
            dets = self.write_results(prediction_bak.clone(), confidence, num_classes, nms, nms_conf)
        return dets

    def write_results(self, prediction, confidence, num_classes, nms=True, nms_conf=0.4):
        args = self.detector_opt
        conf_mask = (prediction[:, :, 4] > confidence).float().float().unsqueeze(2)
        prediction = (prediction * conf_mask)
        try:
            ind_nz = jt.nonzero(prediction[:, :, 4]).transpose(0, 1)
        except:
            return 0
        box_a = jt.rand(*prediction.shape)
        box_a[:, :, 0] = (prediction[:, :, 0] - (prediction[:, :, 2] / 2))
        box_a[:, :, 1] = (prediction[:, :, 1] - (prediction[:, :, 3] / 2))
        box_a[:, :, 2] = (prediction[:, :, 0] + (prediction[:, :, 2] / 2))
        box_a[:, :, 3] = (prediction[:, :, 1] + (prediction[:, :, 3] / 2))
        prediction[:, :, :4] = box_a[:, :, :4]
        batch_size = prediction.shape[0]
        output = jt.rand(1, (prediction.shape[2] + 1))
        write = False
        num = 0
        for ind in range(batch_size):
            image_pred = prediction[ind]
            (max_conf_score, max_conf) = jt.argmax(image_pred[:, 5:(5 + num_classes)], dim=1)
            max_conf = max_conf.float().unsqueeze(1)
            max_conf_score = max_conf_score.float().unsqueeze(1)
            seq = (image_pred[:, :5], max_conf, max_conf_score)
            image_pred = jt.contrib.concat(seq, dim=1)
            non_zero_ind = jt.nonzero(image_pred[:, 4])
            image_pred_ = image_pred[non_zero_ind.squeeze(-1), :].view(((- 1), 7))
            try:
                img_classes = unique(image_pred_[:, (- 1)])
            except:
                continue
            for cls in img_classes:
                if (cls != 0):
                    continue
                cls_mask = (image_pred_ * (image_pred_[:, (- 1)] == cls).float().unsqueeze(1))
                class_mask_ind = jt.nonzero(cls_mask[:, (- 2)]).squeeze(-1)
                image_pred_class = image_pred_[class_mask_ind].view(((- 1), 7))
                conf_sort_index = jt.argsort(image_pred_class[:, 4], descending=True)[0]
                image_pred_class = image_pred_class[conf_sort_index]
                idx = image_pred_class.shape[0]
                if nms:
                    if (platform.system() != 'Windows'):
                        nms_op = getattr(nms_wrapper, 'nms')
                        inds = nms_op(image_pred_class[:, :4], image_pred_class[:, 4:5],nms_conf)
                        image_pred_class = image_pred_class[inds]
                    else:
                        max_detections = []
                        while image_pred_class.shape[0]:
                            max_detections.append(image_pred_class[0].unsqueeze(0))
                            if (len(image_pred_class) == 1):
                                break
                            ious = bbox_iou(max_detections[(- 1)], image_pred_class[1:], args)
                            image_pred_class = image_pred_class[1:][(ious < nms_conf)]
                        image_pred_class = jt.contrib.concat(max_detections).data
                batch_ind = jt.full((image_pred_class.size(0), 1), ind)
                seq = (batch_ind, image_pred_class)
                if (not write):
                    output = jt.contrib.concat(seq, dim=1)
                    write = True
                else:
                    out = jt.contrib.concat(seq, dim=1)
                    output = jt.contrib.concat((output, out))
                num += 1
        if (not num):
            return 0
        return output

    def detect_one_img(self, img_name):
        '\n        Detect bboxs in one image\n        Input: \'str\', full path of image\n        Output: \'[{"category_id":1,"score":float,"bbox":[x,y,w,h],"image_id":str},...]\',\n        The output results are similar with coco results type, except that image_id uses full path str\n        instead of coco %012d id for generalization. \n        '
        args = self.detector_opt
        _CUDA = True
        if args:
            if (args.gpus[0] < 0):
                _CUDA = False
        if (not self.model):
            self.load_model()
        # if isinstance(self.model, torch.nn.DataParallel):
        #     self.model = self.model.module
        dets_results = []
        (img, orig_img, img_dim_list) = prep_image(img_name, self.inp_dim)
        with jt.no_grad():
            img_dim_list = jt.float32([img_dim_list]).repeat(1, 2)
            # img = (img.to(args.device) if args else img.cuda())
            prediction = self.model(img, args=args)
            dets = self.dynamic_write_results(prediction, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thres)
            if (isinstance(dets, int) or (dets.shape[0] == 0)):
                return None
            # dets = dets.cpu()
            img_dim_list = img_dim_list[dets[:, 0].long()]
            scaling_factor = jt.min((self.inp_dim / img_dim_list), 1).view((- 1), 1)
            dets[:, [1, 3]] -= ((self.inp_dim - (scaling_factor * img_dim_list[:, 0].view(((- 1), 1)))) / 2)
            dets[:, [2, 4]] -= ((self.inp_dim - (scaling_factor * img_dim_list[:, 1].view(((- 1), 1)))) / 2)
            dets[:, 1:5] /= scaling_factor
            for i in range(dets.shape[0]):
                dets[(i, [1, 3])] = jt.clamp(dets[(i, [1, 3])], min_v=0.0, max_v=img_dim_list[(i, 0)])
                dets[(i, [2, 4])] = jt.clamp(dets[(i, [2, 4])], min_v=0.0, max_v=img_dim_list[(i, 1)])
                det_dict = {}
                x = float(dets[(i, 1)])
                y = float(dets[(i, 2)])
                w = float((dets[(i, 3)] - dets[(i, 1)]))
                h = float((dets[(i, 4)] - dets[(i, 2)]))
                det_dict['category_id'] = 1
                det_dict['score'] = float(dets[(i, 5)])
                det_dict['bbox'] = [x, y, w, h]
                det_dict['image_id'] = int(os.path.basename(img_name).split('.')[0])
                dets_results.append(det_dict)
            return dets_results
