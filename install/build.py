import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['orn/src/liborn.c']
headers = ['orn/src/liborn.h']
extra_objects = []
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['orn/src/liborn_cuda.c']
    headers += ['orn/src/liborn_cuda.h']
    extra_objects += ['orn/src/liborn_kernel.cu.o']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

ffi = create_extension(
    'orn._ext.liborn',
    package=True,
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    include_dirs=['orn/src'],
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    extra_compile_args=['-fopenmp'],
)

if __name__ == '__main__':
    ffi.build()
