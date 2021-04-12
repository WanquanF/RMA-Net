from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='point_render',
    ext_modules=[
        CUDAExtension('point_render',[
            'point_render.cpp',
            'point_render_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)