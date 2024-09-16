from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='droid_backends',
    ext_modules=[
        CUDAExtension('droid_backends',
            include_dirs=[osp.join(ROOT, 'thirdparty/eigen')],
            sources=[
                'src/droid.cpp', 
                'src/droid_kernels.cu'
            ],
            extra_compile_args={
                'cxx': ['-O0', '-g'],
                'nvcc': ['-O0', '-g', '-G', '-Xcompiler=-fopenmp',
                    '-gencode=arch=compute_86,code=sm_86',
                ]
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)


