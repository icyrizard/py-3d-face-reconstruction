import numpy as np
from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize
extensions = [
    Extension(
        'generate_head_texture',
        ['generate_head_texture.pyx'],
        include_dirs=[np.get_include()], )
]

setup(
    ext_modules=cythonize(extensions),
)
