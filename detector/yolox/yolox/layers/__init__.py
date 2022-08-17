
import jittor as jt
from jittor import init
from jittor import nn
from .jit_ops import FastCOCOEvalOp, JitOp
try:
    from .fast_coco_eval_api import COCOeval_opt
except ImportError:
    pass
