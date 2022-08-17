
import jittor as jt
from jittor import init
from jittor import nn
import glob
import importlib
import os
import sys
import time
from typing import List
__all__ = ['JitOp', 'FastCOCOEvalOp']

class JitOp():
    '\n    Just-in-time compilation of ops.\n\n    Some code of `JitOp` is inspired by `deepspeed.op_builder`,\n    check the following link for more details:\n    https://github.com/microsoft/DeepSpeed/blob/master/op_builder/builder.py\n    '

    def __init__(self, name):
        self.name = name

    def absolute_name(self) -> str:
        'Get absolute build path for cases where the op is pre-installed.'
        pass

    def sources(self) -> List:
        'Get path list of source files of op.\n\n        NOTE: the path should be elative to root of package during building,\n            Otherwise, exception will be raised when building package.\n            However, for runtime building, path will be absolute.\n        '
        pass

    def include_dirs(self) -> List:
        '\n        Get list of include paths, relative to root of package.\n\n        NOTE: the path should be elative to root of package.\n            Otherwise, exception will be raised when building package.\n        '
        return []

    def define_macros(self) -> List:
        'Get list of macros to define for op'
        return []

    def cxx_args(self) -> List:
        'Get optional list of compiler flags to forward'
        args = (['-O2'] if (sys.platform == 'win32') else ['-O3', '-std=c++14', '-g', '-Wno-reorder'])
        return args

    def nvcc_args(self) -> List:
        'Get optional list of compiler flags to forward to nvcc when building CUDA sources'
        args = ['-O3', '--use_fast_math', ('-std=c++17' if (sys.platform == 'win32') else '-std=c++14'), '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']
        return args

    def build_op(self):
        return CppExtension(name=self.absolute_name(), sources=self.sources(), include_dirs=self.include_dirs(), define_macros=self.define_macros(), extra_compile_args={'cxx': self.cxx_args()})

    def load(self, verbose=True):
        try:
            return importlib.import_module(self.absolute_name())
        except Exception:
            from yolox.utils import wait_for_the_master
            with wait_for_the_master():
                return self.jit_load(verbose)

    def jit_load(self, verbose=True):
        from loguru import logger
        try:
            import ninja
        except ImportError:
            if verbose:
                logger.warning(f'Ninja is not installed, fall back to normal installation for {self.name}.')
        build_tik = time.time()
        op_module = load(name=self.name, sources=self.sources(), extra_cflags=self.cxx_args(), extra_cuda_cflags=self.nvcc_args(), verbose=verbose)
        build_duration = (time.time() - build_tik)
        if verbose:
            logger.info(f'Load {self.name} op in {build_duration:.3f}s.')
        return op_module

    def clear_dynamic_library(self):
        'Remove dynamic libraray files generated by JIT compilation.'
        module = 
        raise RuntimeError('origin source: <self.load()>, There are needed 3 args in Pytorch load function, but you only provide 0')
        
        raise RuntimeError('original source: <os.remove(module.__file__)>, remove is not supported in Jittor yet. We will appreciate it if you provide an implementation of remove and make pull request at https://github.com/Jittor/jittor.')

class FastCOCOEvalOp(JitOp):

    def __init__(self, name='fast_cocoeval'):
        super().__init__(name=name)

    def absolute_name(self):
        return f'yolox.layers.{self.name}'

    def sources(self):
        sources = glob.glob(os.path.join('yolox', 'layers', 'cocoeval', '*.cpp'))
        if (not sources):
            import yolox
            code_path = os.path.join(yolox.__path__[0], 'layers', 'cocoeval', '*.cpp')
            sources = glob.glob(code_path)
        return sources

    def include_dirs(self):
        return [os.path.join('yolox', 'layers', 'cocoeval')]
