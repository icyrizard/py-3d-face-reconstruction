HERE := $(shell pwd)
TARGETS += src/reconstruction/texture.so

ifeq ($(OS),Darwin)
HALIDE_LINK:=https://github.com/halide/Halide/releases/download/release_2016_04_27/halide-mac-64-trunk-2f11b9fce62f596e832907b82d87e8f75c53dd07.tgz
else
# todo link for linux, depends on gcc version, no check is in place for the
# right gcc version
HALIDE_LINK:=https://github.com/halide/Halide/releases/download/release_2016_04_27/halide-linux-64-gcc53-trunk-2f11b9fce62f596e832907b82d87e8f75c53dd07.tgz
endif

texture.so: src/reconstruction/texture.pyx
	(cd src/reconstruction; python setup.py build_ext --inplace)

halide_2016_04_27.tar.gz:
	wget -O data/halide_2016_04_27.tar.gz $(HALIDE_LINK)

vendor/halide:
	mkdir -p vendor/
	tar -xzvf data/halide_2016_04_27.tar.gz -C vendor/

halide: vendor/halide data/halide_2016_04_27.tar.gz

#DYLD_LIBRARY_PATH=vendor/halide/bin ./lesson_01

src/reconstruction/texture_halide:
	clang++ src/reconstruction/texture_halide.cpp -g -I vendor/halide/include -L vendor/halide/bin -lHalide -o $@ -std=c++11
