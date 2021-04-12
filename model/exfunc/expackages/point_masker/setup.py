from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension



setup(
    name='point_masker',
    ext_modules=[
        CUDAExtension('point_masker',[
            'point_masker.cpp',
            'point_masker_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
