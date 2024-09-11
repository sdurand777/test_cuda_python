

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os


setup(
    name='cuda_add',
    ext_modules=[
        CUDAExtension('cuda_add',
            sources=[
                'src/cuda_add.cpp',
                'src/cuda_add.cu'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-G',
                    '-gencode=arch=compute_60,code=sm_60',
                    '-gencode=arch=compute_61,code=sm_61',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                ]
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)




# # Compilation flags for CUDA
# cuda_flags = [
#     '-std=c++11',
#     '-lcudart',
#     '-lcuda',
#     '-lcufft',
#     '-lcublas',
#     '-lcurand',
# ]
#
# # Path to nvcc compiler
# nvcc_path = '/usr/local/cuda/bin/nvcc'
#
# ext_modules = [
#     Pybind11Extension(
#         'cuda_add',
#         ['src/cuda_add.cu'],
#         extra_compile_args={'cxx': ['-O3'], 'nvcc': cuda_flags},
#         language='c++',
#     ),
# ]
#
# setup(
#     name='cuda_add',
#     ext_modules=ext_modules,
#     cmdclass={'build_ext': build_ext},
# )
