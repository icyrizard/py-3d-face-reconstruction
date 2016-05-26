VIRTUALENV := venv
PYTHON := python2.7
PYTHON_BIN_PATH := /usr/local/bin/$(PYTHON)
SITE_PACKAGES := $(VIRTUALENV)/lib/$(PYTHON)/site-packages

OPENCV:= $(SITE_PACKAGES)/cv.py $(SITE_PACKAGES)/cv2.so

TARGETS:= $(OPENCV) $(VIRTUALENV) data utils
all: $(TARGETS)

include actions.mk

data: data/imm_face_db
utils: generate_head_texture.so

generate_head_texture.so: src/utils/generate_head_texture.pyx
	(cd src/utils; python setup.py build_ext --inplace)


build: requirements.txt
	@(source $(VIRTUALENV)/bin/activate; \
		pip install -r requirements.txt; \
	);

$(VIRTUALENV):
	virtualenv -p $(PYTHON_BIN_PATH) venv

$(SITE_PACKAGES)/cv%:
	@/bin/ln -s `scripts/get_site_package_location.sh`/$(shell basename $@) $@
	@ls $@
