
import jittor as jt
from jittor import init
from jittor import nn
import os
sources = []
headers = []
defines = []
with_cuda = False
if jt.has_cuda:
    print('Including CUDA code.')
    sources += ['src/psroi_pooling_cuda.c']
    headers += ['src/psroi_pooling_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True
this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = ['src/cuda/psroi_pooling.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]
ffi = create_extension('_ext.psroi_pooling', headers=headers, sources=sources, define_macros=defines, relative_to=__file__, with_cuda=with_cuda, extra_objects=extra_objects)
if (__name__ == '__main__'):
    ffi.build()
