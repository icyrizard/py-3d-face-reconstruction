#!/bin/bash

rm -rf build/
mkdir -p build

(cd build; \
    cmake .. \
        -DUSE_PYTHON=2 \
        -DHALIDE_LIBRARIES=../../build/bin/libHalide.so \
        -DHALIDE_INCLUDE_DIR=../../build/include/ \
        -DHALIDE_ROOT_DIR=../../build/ \
        -DPYTHON_LIBRARY=/usr/local/Cellar/python/2.7.11/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib;\
    make -j8
)
