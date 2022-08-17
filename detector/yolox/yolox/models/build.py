
import jittor as jt
from jittor import init
from jittor import nn
__all__ = ['create_yolox_model', 'yolox_nano', 'yolox_tiny', 'yolox_s', 'yolox_m', 'yolox_l', 'yolox_x', 'yolov3']
_CKPT_ROOT_URL = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download'
_CKPT_FULL_PATH = {'yolox-nano': f'{_CKPT_ROOT_URL}/0.1.1rc0/yolox_nano.pth', 'yolox-tiny': f'{_CKPT_ROOT_URL}/0.1.1rc0/yolox_tiny.pth', 'yolox-s': f'{_CKPT_ROOT_URL}/0.1.1rc0/yolox_s.pth', 'yolox-m': f'{_CKPT_ROOT_URL}/0.1.1rc0/yolox_m.pth', 'yolox-l': f'{_CKPT_ROOT_URL}/0.1.1rc0/yolox_l.pth', 'yolox-x': f'{_CKPT_ROOT_URL}/0.1.1rc0/yolox_x.pth', 'yolov3': f'{_CKPT_ROOT_URL}/0.1.1rc0/yolox_darknet.pth'}

def create_yolox_model(name: str, pretrained: bool=True, num_classes: int=80,
                       device=None
                       ) -> nn.Module:
    'creates and loads a YOLOX model\n\n    Args:\n        name (str): name of model. for example, "yolox-s", "yolox-tiny".\n        pretrained (bool): load pretrained weights into the model. Default to True.\n        num_classes (int): number of model classes. Defalut to 80.\n        device (str): default device to for model. Defalut to None.\n\n    Returns:\n        YOLOX model (nn.Module)\n    '
    from yolox.exp import get_exp, Exp
    if (device is None):
        device = ('cuda:0' if jt.has_cuda else 'cpu')
    # device = torch.device(device)
    assert (name in _CKPT_FULL_PATH), f'user should use one of value in {_CKPT_FULL_PATH.keys()}'
    exp: Exp = get_exp(exp_name=name)
    exp.num_classes = num_classes
    yolox_model = exp.get_model()
    if (pretrained and (num_classes == 80)):
        weights_url = _CKPT_FULL_PATH[name]
        ckpt = jt.load(weights_url)
        if ('model' in ckpt):
            ckpt = ckpt['model']
        yolox_model.load_parameters(ckpt)
    # yolox_model.to(device)
    return yolox_model

def yolox_nano(pretrained=True, num_classes=80,
               device=None
               ):
    return create_yolox_model('yolox-nano', pretrained, num_classes, device)

def yolox_tiny(pretrained=True, num_classes=80, device=None):
    return create_yolox_model('yolox-tiny', pretrained, num_classes, device)

def yolox_s(pretrained=True, num_classes=80, device=None):
    return create_yolox_model('yolox-s', pretrained, num_classes, device)

def yolox_m(pretrained=True, num_classes=80, device=None):
    return create_yolox_model('yolox-m', pretrained, num_classes, device)

def yolox_l(pretrained=True, num_classes=80, device=None):
    return create_yolox_model('yolox-l', pretrained, num_classes, device)

def yolox_x(pretrained=True, num_classes=80, device=None):
    return create_yolox_model('yolox-x', pretrained, num_classes, device)

def yolov3(pretrained=True, num_classes=80, device=None):
    return create_yolox_model('yolox-tiny', pretrained, num_classes, device)
