
import jittor as jt
from jittor import init
import math
from jittor import nn
from detector.yolox.yolox.utils import bboxes_iou, meshgrid
from .losses import IOUloss
from .network_blocks import BaseConv, DWConv
from jittor_implementations import one_hot

class YOLOXHead(nn.Module):

    def __init__(self, num_classes, width=1.0, strides=[8, 16, 32], in_channels=[256, 512, 1024], act='silu', depthwise=False):
        super().__init__()
        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = (DWConv if depthwise else BaseConv)
        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels=int((in_channels[i] * width)), out_channels=int((256 * width)), ksize=1, stride=1, act=act))
            self.cls_convs.append(nn.Sequential(*[Conv(in_channels=int((256 * width)), out_channels=int((256 * width)), ksize=3, stride=1, act=act), Conv(in_channels=int((256 * width)), out_channels=int((256 * width)), ksize=3, stride=1, act=act)]))
            self.reg_convs.append(nn.Sequential(*[Conv(in_channels=int((256 * width)), out_channels=int((256 * width)), ksize=3, stride=1, act=act), Conv(in_channels=int((256 * width)), out_channels=int((256 * width)), ksize=3, stride=1, act=act)]))
            self.cls_preds.append(nn.Conv(int((256 * width)), (self.n_anchors * self.num_classes), 1, stride=1, padding=0))
            self.reg_preds.append(nn.Conv(int((256 * width)), 4, 1, stride=1, padding=0))
            self.obj_preds.append(nn.Conv(int((256 * width)), (self.n_anchors * 1), 1, stride=1, padding=0))
        self.use_l1 = False
        self.l1_loss = nn.L1Loss()
        self.bcewithlog_loss = nn.BCEWithLogitsLoss()
        self.iou_loss = IOUloss(reduction='none')
        self.strides = strides
        self.grids = ([jt.zeros(1)] * len(in_channels))

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view((self.n_anchors, (- 1)))
            b = jt.full_like(b, (- math.log(((1 - prior_prob) / prior_prob))))
            conv.bias = jt.array(b.view((- 1)))
        for conv in self.obj_preds:
            b = conv.bias.view((self.n_anchors, (- 1)))
            b = jt.full_like(b, (- math.log(((1 - prior_prob) / prior_prob))))
            conv.bias = jt.array(b.view((- 1)))

    def execute(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        for (k, (cls_conv, reg_conv, stride_this_level, x)) in enumerate(zip(self.cls_convs, self.reg_convs, self.strides, xin)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x
            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            if self.is_train:
                output = jt.contrib.concat([reg_output, obj_output, cls_output], dim=1)
                (output, grid) = self.get_output_and_grid(output, k, stride_this_level, xin[0].dtype)
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(jt.zeros((1, grid.shape[1])).fill_(stride_this_level).type_as(xin[0]))
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    (hsize, wsize) = reg_output.shape[(- 2):]
                    reg_output = reg_output.view((batch_size, self.n_anchors, 4, hsize, wsize))
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape((batch_size, (- 1), 4))
                    origin_preds.append(reg_output.clone())
            else:
                output = jt.contrib.concat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], dim=1)
            outputs.append(output)
        if self.is_train:
            return self.get_losses(imgs, x_shifts, y_shifts, expanded_strides, labels, jt.contrib.concat(outputs, dim=1), origin_preds, dtype=xin[0].dtype)
        else:
            self.hw = [x.shape[(- 2):] for x in outputs]
            outputs = jt.concat([x.flatten(start_dim=2) for x in outputs], dim=2).permute((0, 2, 1))
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].dtype)
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]
        batch_size = output.shape[0]
        n_ch = (5 + self.num_classes)
        (hsize, wsize) = output.shape[(- 2):]
        if (grid.shape[2:4] != output.shape[2:4]):
            (yv, xv) = meshgrid([jt.arange(hsize), jt.arange(wsize)])
            grid = jt.stack((xv, yv), 2).view((1, 1, hsize, wsize, 2)).astype(dtype)
            self.grids[k] = grid
        output = output.view((batch_size, self.n_anchors, n_ch, hsize, wsize))
        output = output.permute(0, 1, 3, 4, 2).reshape((batch_size, ((self.n_anchors * hsize) * wsize), (- 1)))
        grid = grid.view((1, (- 1), 2))
        output[..., :2] = ((output[..., :2] + grid) * stride)
        output[..., 2:4] = (jt.exp(output[..., 2:4]) * stride)
        return (output, grid)

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for ((hsize, wsize), stride) in zip(self.hw, self.strides):
            (yv, xv) = meshgrid([jt.arange(hsize), jt.arange(wsize)])
            grid = jt.stack((xv, yv), 2).view((1, (- 1), 2))
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(jt.full((*shape, 1), stride))
        grids = jt.contrib.concat(grids, dim=1).astype(dtype)
        strides = jt.contrib.concat(strides, dim=1).astype(dtype)
        outputs[..., :2] = ((outputs[..., :2] + grids) * strides)
        outputs[..., 2:4] = (jt.exp(outputs[..., 2:4]) * strides)
        return outputs

    def get_losses(self, imgs, x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype):
        from loguru import logger
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4].unsqueeze((- 1))
        cls_preds = outputs[:, :, 5:]
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
        total_num_anchors = outputs.shape[1]
        x_shifts = jt.contrib.concat(x_shifts, dim=1)
        y_shifts = jt.contrib.concat(y_shifts, dim=1)
        expanded_strides = jt.contrib.concat(expanded_strides, dim=1)
        if self.use_l1:
            origin_preds = jt.contrib.concat(origin_preds, dim=1)
        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        num_fg = 0.0
        num_gts = 0.0
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if (num_gt == 0):
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                try:
                    (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img) = self.get_assignments(batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts, cls_preds, bbox_preds, obj_preds, labels, imgs)
                except RuntimeError as e:
                    if ('CUDA out of memory. ' not in str(e)):
                        raise
                    logger.error('OOM RuntimeError is raised due to the huge memory cost during label assignment.                            CPU mode is applied in this batch. If you want to avoid this issue,                            try to reduce the batch size or image size.')
                    # torch.cuda.empty_cache()
                    jt.gc()
                    (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img) = self.get_assignments(batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts, cls_preds, bbox_preds, obj_preds, labels, imgs, 'cpu')
                num_fg += num_fg_img
                cls_target = (one_hot(gt_matched_classes.astype(jt.int64), self.num_classes) * pred_ious_this_matching.unsqueeze((- 1)))
                obj_target = fg_mask.unsqueeze((- 1))
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(outputs.new_zeros((num_fg_img, 4)), gt_bboxes_per_image[matched_gt_inds], expanded_strides[0][fg_mask], x_shifts=x_shifts[0][fg_mask], y_shifts=y_shifts[0][fg_mask])
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.astype(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)
        cls_targets = jt.contrib.concat(cls_targets, dim=0)
        reg_targets = jt.contrib.concat(reg_targets, dim=0)
        obj_targets = jt.contrib.concat(obj_targets, dim=0)
        fg_masks = jt.contrib.concat(fg_masks, dim=0)
        if self.use_l1:
            l1_targets = jt.contrib.concat(l1_targets, dim=0)
        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view((- 1), 4)[fg_masks], reg_targets).sum() / num_fg)
        loss_obj = (self.bcewithlog_loss(obj_preds.view((- 1), 1), obj_targets).sum() / num_fg)
        loss_cls = (self.bcewithlog_loss(cls_preds.view((- 1), self.num_classes)[fg_masks], cls_targets).sum() / num_fg)
        if self.use_l1:
            loss_l1 = (self.l1_loss(origin_preds.view((- 1), 4)[fg_masks], l1_targets).sum() / num_fg)
        else:
            loss_l1 = 0.0
        reg_weight = 5.0
        loss = ((((reg_weight * loss_iou) + loss_obj) + loss_cls) + loss_l1)
        return (loss, (reg_weight * loss_iou), loss_obj, loss_cls, loss_l1, (num_fg / max(num_gts, 1)))

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-08):
        l1_target[:, 0] = ((gt[:, 0] / stride) - x_shifts)
        l1_target[:, 1] = ((gt[:, 1] / stride) - y_shifts)
        l1_target[:, 2] = jt.log(((gt[:, 2] / stride) + eps))
        l1_target[:, 3] = jt.log(((gt[:, 3] / stride) + eps))
        return l1_target

    @jt.no_grad()
    def get_assignments(self, batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts, cls_preds, bbox_preds, obj_preds, labels, imgs, mode='gpu'):
        if (mode == 'cpu'):
            print('------------CPU Mode for This Batch-------------')
            gt_bboxes_per_image = gt_bboxes_per_image.float()
            bboxes_preds_per_image = bboxes_preds_per_image.float()
            gt_classes = gt_classes.float()
            expanded_strides = expanded_strides.float()
            x_shifts = x_shifts
            y_shifts = y_shifts
        (fg_mask, is_in_boxes_and_center) = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt)
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]
        if (mode == 'cpu'):
            gt_bboxes_per_image = gt_bboxes_per_image
            bboxes_preds_per_image = bboxes_preds_per_image
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        gt_cls_per_image = one_hot(gt_classes.astype(jt.int64), self.num_classes).float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)
        pair_wise_ious_loss = (- jt.log((pair_wise_ious + 1e-08)))
        if (mode == 'cpu'):
            (cls_preds_, obj_preds_) = (cls_preds_, obj_preds_)
        with jt.cuda.amp.autocast(enabled=False):
            # cls_preds_ = (cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_())
            # pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction='none').sum(dim=(- 1))
            cls_preds_ = (cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1)* obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1))
            pair_wise_cls_loss = jt.nn.binary_cross_entropy_with_logits(cls_preds_.sqrt_(), gt_cls_per_image, reduction='none').sum(dim=(- 1))
        del cls_preds_
        cost = ((pair_wise_cls_loss + (3.0 * pair_wise_ious_loss)) + (100000.0 * (~ is_in_boxes_and_center)))
        (num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        if (mode == 'cpu'):
            gt_matched_classes = gt_matched_classes
            fg_mask = fg_mask
            pred_ious_this_matching = pred_ious_this_matching
            matched_gt_inds = matched_gt_inds
        return (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg)

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = (x_shifts[0] * expanded_strides_per_image)
        y_shifts_per_image = (y_shifts[0] * expanded_strides_per_image)
        x_centers_per_image = (x_shifts_per_image + (0.5 * expanded_strides_per_image)).unsqueeze(0).repeat(num_gt, 1)
        y_centers_per_image = (y_shifts_per_image + (0.5 * expanded_strides_per_image)).unsqueeze(0).repeat(num_gt, 1)
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0] - (0.5 * gt_bboxes_per_image[:, 2])).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0] + (0.5 * gt_bboxes_per_image[:, 2])).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1] - (0.5 * gt_bboxes_per_image[:, 3])).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1] + (0.5 * gt_bboxes_per_image[:, 3])).unsqueeze(1).repeat(1, total_num_anchors)
        b_l = (x_centers_per_image - gt_bboxes_per_image_l)
        b_r = (gt_bboxes_per_image_r - x_centers_per_image)
        b_t = (y_centers_per_image - gt_bboxes_per_image_t)
        b_b = (gt_bboxes_per_image_b - y_centers_per_image)
        bbox_deltas = jt.stack([b_l, b_t, b_r, b_b], dim=2)
        is_in_boxes = (bbox_deltas.min(dim=(- 1)).values > 0.0)
        is_in_boxes_all = (is_in_boxes.sum(dim=0) > 0)
        center_radius = 2.5
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0].unsqueeze(1).repeat(1, total_num_anchors) - (center_radius * expanded_strides_per_image.unsqueeze(0)))
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0].unsqueeze(1).repeat(1, total_num_anchors) + (center_radius * expanded_strides_per_image.unsqueeze(0)))
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1].unsqueeze(1).repeat(1, total_num_anchors) - (center_radius * expanded_strides_per_image.unsqueeze(0)))
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1].unsqueeze(1).repeat(1, total_num_anchors) + (center_radius * expanded_strides_per_image.unsqueeze(0)))
        c_l = (x_centers_per_image - gt_bboxes_per_image_l)
        c_r = (gt_bboxes_per_image_r - x_centers_per_image)
        c_t = (y_centers_per_image - gt_bboxes_per_image_t)
        c_b = (gt_bboxes_per_image_b - y_centers_per_image)
        center_deltas = jt.stack([c_l, c_t, c_r, c_b], dim=2)
        is_in_centers = (center_deltas.min(dim=(- 1)).values > 0.0)
        is_in_centers_all = (is_in_centers.sum(dim=0) > 0)
        is_in_boxes_anchor = (is_in_boxes_all | is_in_centers_all)
        is_in_boxes_and_center = (is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor])
        return (is_in_boxes_anchor, is_in_boxes_and_center)

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = jt.zeros_like(cost)
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.shape[1])
        (topk_ious, _) = jt.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = jt.clamp(topk_ious.sum(dim=1).int(), min_v=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            (_, pos_idx) = jt.topk(cost[gt_idx], dynamic_ks[gt_idx], largest=False)
            matching_matrix[gt_idx][pos_idx] = 1
        del topk_ious, dynamic_ks, pos_idx
        anchor_matching_gt = matching_matrix.sum(dim=0)
        if ((anchor_matching_gt > 1).sum() > 0):
            (_, cost_argmin) = jt.min(cost[:, (anchor_matching_gt > 1)], dim=0)
            matching_matrix[:, (anchor_matching_gt > 1)] *= 0
            matching_matrix[(cost_argmin, (anchor_matching_gt > 1))] = 1
        fg_mask_inboxes = (matching_matrix.sum(dim=0) > 0)
        num_fg = fg_mask_inboxes.sum().item()
        fg_mask[fg_mask.clone()] = fg_mask_inboxes
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(dim=0)[fg_mask_inboxes]
        return (num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds)
