VIRTUALENV := venv
PYTHON := python2.7
PYTHON_BIN_PATH := /usr/local/bin/$(PYTHON)
SITE_PACKAGES := $(VIRTUALENV)/lib/$(PYTHON)/site-packages

OPENCV:= $(SITE_PACKAGES)/cv.py $(SITE_PACKAGES)/cv2.so

TARGETS:= $(OPENCV) $(VIRTUALENV) data reconstruction
all: $(TARGETS)

include actions.mk
include src/reconstruction/build.mk

data: data/imm_face_db
reconstruction: texture.so src/reconstruction/texture_halide

OS := $(shell uname)

build: requirements.txt
	@(source $(VIRTUALENV)/bin/activate; \
		pip install -r requirements.txt; \
	);

$(VIRTUALENV):
	virtualenv -p $(PYTHON_BIN_PATH) venv

$(SITE_PACKAGES)/cv%:
	@/bin/ln -s `scripts/get_site_package_location.sh`/$(shell basename $@) $@
	@ls $@
