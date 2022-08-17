
import jittor as jt
from jittor import init
from jittor import nn
import os
from collections import OrderedDict
try:
except ImportError:

def load_checkpoint(model, checkpoint_path):
    if (checkpoint_path and os.path.isfile(checkpoint_path)):
        print("=> Loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = jt.load(checkpoint_path)
        if (isinstance(checkpoint, dict) and ('state_dict' in checkpoint)):
            new_state_dict = OrderedDict()
            for (k, v) in checkpoint['state_dict'].items():
                if k.startswith('module'):
                    name = k[7:]
                else:
                    name = k
                new_state_dict[name] = v
            model.load_parameters(new_state_dict)
        else:
            model.load_parameters(checkpoint)
        print("=> Loaded checkpoint '{}'".format(checkpoint_path))
    else:
        print("=> Error: No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

def load_pretrained(model, url, filter_fn=None, strict=True):
    if (not url):
        print('=> Warning: Pretrained model URL is empty, using random initialization.')
        return
    state_dict = load_state_dict_from_url(url, progress=False, map_location='cpu')
    if (filter_fn is not None):
        state_dict = filter_fn(state_dict)
    model.load_parameters(state_dict, strict=strict)
