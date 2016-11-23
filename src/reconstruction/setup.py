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
    Extension(
        'fit',
        ['fit-model.cpp'],
        language="c++",
        include_dirs=[
            '/usr/local/eos/include/',  # path need to be changed in future
            '/usr/local/eos/3rdparty/glm/',
            '/usr/local/eos/3rdparty/cereal-1.1.1/include/',
            '/usr/local/include/opencv2/',
            '/usr/include/boost/'
        ],
        library_dirs=[
            '/usr/local/eos/bin',
            '/usr/lib/x86_64-linux-gnu/',
            '/usr/local/lib/'
        ],
        libraries=[
            'boost_program_options',
            'boost_filesystem',
            'opencv_world'
        ],
        extra_compile_args=['-std=c++14'], )
        #include_dirs=[np.get_include()], ),
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
