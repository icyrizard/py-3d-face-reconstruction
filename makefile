#VIRTUALENV := venv
#PYTHON := python2.7
#PYTHON_BIN_PATH := /usr/local/bin/$(PYTHON)
#SITE_PACKAGES := $(VIRTUALENV)/lib/$(PYTHON)/site-packages
#OPENCV := $(SITE_PACKAGES)/cv.py $(SITE_PACKAGES)/cv2.so
#TARGETS := $(OPENCV) $(VIRTUALENV) data reconstruction

VERSION:=v0.1
IMAGE_TAG:= icyrizard/face-reconstruction.git:$(VERSION)
DEBUG:=1
BASE_DOCKER_CMD:= docker run \
	--rm \
	--volume /Users/richard/Documents/sighthub/face-reconstruction/data:/data \
	--volume /Users/richard/Documents/sighthub/face-reconstruction/src:/src \
	-e "DEBUG=$(DEBUG)" \
	$(IMAGE_TAG)

include actions.mk
include src/reconstruction/build.mk

all: $(TARGETS)

data: data/imm_face_db
reconstruction: texture.so

OS := $(shell uname)

build: requirements.txt
	docker build -t $(IMAGE_TAG) .

#@(source $(VIRTUALENV)/bin/activate; \
#	pip install -r requirements.txt; \
#);

run-bash:
	$(BASE_DOCKER_CMD) --interactive --tty $(IMAGE_TAG) /bin/bash

$(VIRTUALENV):
	virtualenv -p $(PYTHON_BIN_PATH) venv

$(SITE_PACKAGES)/cv%:
	@/bin/ln -s `scripts/get_site_package_location.sh`/$(shell basename $@) $@
	@ls $@


