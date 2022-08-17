
import jittor as jt
from jittor import init
from jittor import nn

class IOUloss(nn.Module):

    def __init__(self, reduction='none', loss_type='iou'):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def execute(self, pred, target):
        assert (pred.shape[0] == target.shape[0])
        pred = pred.view(((- 1), 4))
        target = target.view(((- 1), 4))
        tl = jt.max((pred[:, :2] - (pred[:, 2:] / 2)), dim=(target[:, :2] - (target[:, 2:] / 2)))
        br = jt.min((pred[:, :2] + (pred[:, 2:] / 2)), dim=(target[:, :2] + (target[:, 2:] / 2)))
        area_p = jt.prod(pred[:, 2:], dim=1)
        area_g = jt.prod(target[:, 2:], dim=1)
        en = (tl < br).astype(tl.dtype).prod(dim=1)
        area_i = (jt.prod((br - tl), dim=1) * en)
        area_u = ((area_p + area_g) - area_i)
        iou = (area_i / (area_u + 1e-16))
        if (self.loss_type == 'iou'):
            loss = (1 - (iou ** 2))
        elif (self.loss_type == 'giou'):
            c_tl = jt.min((pred[:, :2] - (pred[:, 2:] / 2)), dim=(target[:, :2] - (target[:, 2:] / 2)))
            c_br = jt.max((pred[:, :2] + (pred[:, 2:] / 2)), dim=(target[:, :2] + (target[:, 2:] / 2)))
            area_c = jt.prod((c_br - c_tl), dim=1)
            giou = (iou - ((area_c - area_u) / area_c.clamp(min_v=1e-16)))
            loss = (1 - giou.clamp(min_v=(- 1.0), max_v=1.0))
        if (self.reduction == 'mean'):
            loss = loss.mean()
        elif (self.reduction == 'sum'):
            loss = loss.sum()
        return loss
