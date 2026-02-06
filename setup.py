from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import torch
import os

ext_type = CUDAExtension if torch.cuda.is_available() else CppExtension

lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
setup(
    name='my_custom_ops',
    ext_modules=[
        ext_type(
            name='my_custom_ops_lib',
            sources=['MyScatterNode.cpp'],
            extra_compile_args={'cxx': ['-O3']},
            extra_link_args=[
                f"-Wl,-rpath,{lib_dir} "
            ],
            
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)