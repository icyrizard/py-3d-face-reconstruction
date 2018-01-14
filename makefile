DEBUG:=1
VERSION:=v0.1
IMAGE_TAG:= icyrizard/face-reconstruction.git:$(VERSION)
HERE:=$(shell pwd)
DOCKER_RUN_FLAGS:= --rm \
	--volume $(HERE)/data:/data \
	--volume $(HERE)/src:/src \
	--volume ~/.bash_history:/root/.bash_history \
	-e "DEBUG=$(DEBUG)"

BASE_DOCKER_CMD:= docker run $(DOCKER_RUN_FLAGS) $(IMAGE_TAG)

$(info $(TARGETS))

DEPENDENCIES:= data/imm_face_db
TARGETS:= data/shape_predictor_68_face_landmarks.dat \
	src/reconstruction/texture.so \
	data/imm_face_db \
	data/pca_ibug_shape_model.npy \
	data/pca_ibug_texture_model.npy

all: $(DEPENDENCIES) $(TARGETS)

include actions.mk
include src/reconstruction/build.mk

OS := $(shell uname)

build: requirements.txt
	docker build -t $(IMAGE_TAG) .

run-bash:
	docker run --interactive --tty $(DOCKER_RUN_FLAGS) $(IMAGE_TAG) /bin/bash

run-bash-cmd:
	docker run --interactive --tty $(DOCKER_RUN_FLAGS) $(IMAGE_TAG) \
		/bin/bash -c "$(CMD)"

$(VIRTUALENV):
	virtualenv -p $(PYTHON_BIN_PATH) venv

$(SITE_PACKAGES)/cv%:
	@/bin/ln -s `scripts/get_site_package_location.sh`/$(shell basename $@) $@
	@ls $@


#src/reconstruction/fit.so: src/reconstruction/fit-model.cpp
#	$(BASE_DOCKER_CMD) /bin/bash -c '(cd reconstruction; python setup.py build_ext --inplace)'

#src/reconstruction/fit.so: src/reconstruction/fit-model.cpp
#	$(BASE_DOCKER_CMD) /bin/bash -c \
#		'(cd reconstruction; \
#			clang++ -fPIC -O3 -shared -std=c++14 \
#				-I/usr/local/include/pybind11/include/ \
#				-I/usr/local/eos/include/ \
#				-I/usr/local/eos/3rdparty/glm/ \
#				-I/usr/local/eos/3rdparty/cereal-1.1.1/include/ \
#				-I/usr/local/include/opencv2/ \
#				-I/usr/include/boost/ \
#				-L/usr/local/eos/bin/ \
#				-L/usr/lib/x86_64-linux-gnu/ \
#				-L/usr/local/lib/ \
#				-lboost_program_options \
#				-lboost_filesystem \
#				-lopencv_world \
#				`python-config --cflags --ldflags` \
#			$(notdir $^) -o $(notdir $@) \
#		)'
