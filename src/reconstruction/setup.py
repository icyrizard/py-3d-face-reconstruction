import numpy as np
from distutils.core import setup
from distutils.extension import Extension
import os

os.environ["CXX"] = "g++"
os.environ["CC"] = "g++"

from Cython.Build import cythonize

#g++ src/reconstruction/texture_halide.cpp
# -g -I vendor/halide/include -L vendor/halide/bin -lHalide -o $@ -std=c++11

extensions = [
    Extension(
        'texture',
        ['texture.pyx'],
        include_dirs=[np.get_include()], ),
    #Extension(
    #    'halide',
    #    ['texture_halide.cpp'],
    #    language="c++",
    #    include_dirs=[
    #        '../../vendor/halide/include',
    #        '../../vendor/halide'
    #    ],
    #    library_dirs=['../../vendor/halide/bin'],
    #    libraries=['Halide'],
    #    extra_compile_args=['-std=c++11'],
    #)
]

setup(
    ext_modules=cythonize(extensions),
)
