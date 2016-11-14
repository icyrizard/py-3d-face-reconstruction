DEBUG:=1
VERSION:=v0.1
IMAGE_TAG:= icyrizard/face-reconstruction.git:$(VERSION)
HERE:=$(shell pwd)
DOCKER_RUN_FLAGS:= --rm \
	--volume $(HERE)/data:/data \
	--volume $(HERE)/src:/src \
	-e "DEBUG=$(DEBUG)" \
	-p 6930:8888

BASE_DOCKER_CMD:= docker run $(DOCKER_RUN_FLAGS) $(IMAGE_TAG)

$(info $(TARGETS))

DEPENDENCIES:= data/imm_face_db
TARGETS:= shape_predictor_68_face_landmarks.dat\
	src/reconstruction/texture.so \
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

$(VIRTUALENV):
	virtualenv -p $(PYTHON_BIN_PATH) venv

$(SITE_PACKAGES)/cv%:
	@/bin/ln -s `scripts/get_site_package_location.sh`/$(shell basename $@) $@
	@ls $@


