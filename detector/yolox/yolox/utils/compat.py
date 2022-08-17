
import jittor as jt
from jittor import init
from jittor import nn
# _TORCH_VER = [int(x) for x in jt.__version__.split('.')[:2]]
# __all__ = ['meshgrid']

# def meshgrid(*tensors):
#     if (_TORCH_VER >= [1, 10]):
#         return jt.meshgrid(*tensors)
#     else:
#         return jt.meshgrid(*tensors)


def meshgrid(*tensors):
    return jt.meshgrid(*tensors)