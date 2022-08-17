
import jittor as jt
from jittor import init
from jittor import nn
'\n@author:  tanghy\n@contact: thutanghy@gmail.com\n'

def bn_no_bias(in_features):
    bn_layer = nn.BatchNorm1d(in_features, affine=None)
    bn_layer.bias.requires_grad_(False)
    return bn_layer

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if (classname.find('Linear') != (- 1)):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if (m.bias is not None):
            init.constant_(m.bias, value=0.0)
    elif (classname.find('Conv') != (- 1)):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if (m.bias is not None):
            init.constant_(m.bias, value=0.0)
    elif (classname.find('BatchNorm') != (- 1)):
        if m.affine:
            init.constant_(m.weight, value=1.0)
            init.constant_(m.bias, value=0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if (classname.find('Linear') != (- 1)):
        init.gauss_(m.weight, std=0.001)
        if (m.bias is not None):
            init.constant_(m.bias, value=0.0)

class BNneckLinear(nn.Module):

    def __init__(self, nID):
        super().__init__()
        self._num_classes = nID
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bnneck = bn_no_bias(2048)
        self.bnneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(2048, self._num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def execute(self, features):
        '\n        See :class:`ReIDHeads.forward`.\n        '
        global_features = self.gap(features)
        global_features = global_features.view((global_features.shape[0], (- 1)))
        bn_features = self.bnneck(global_features)
        if (not self.is_train):
            return F.normalize(bn_features)
        pred_class_logits = self.classifier(bn_features)
        return (global_features, pred_class_logits)
