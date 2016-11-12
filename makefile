DEBUG:=1
VERSION:=v0.1
IMAGE_TAG:= icyrizard/face-reconstruction.git:$(VERSION)
BASE_DOCKER_CMD:= docker run \
	--rm \
	--volume /Users/richard/Documents/sighthub/face-reconstruction/data:/data \
	--volume /Users/richard/Documents/sighthub/face-reconstruction/src:/src \
	-e "DEBUG=$(DEBUG)" \
	-p 8888:8888 \
	$(IMAGE_TAG)

include actions.mk
include src/reconstruction/build.mk

all: $(TARGETS)

data: data/imm_face_db
reconstruction: texture.so

OS := $(shell uname)

build: requirements.txt
	docker build -t $(IMAGE_TAG) .

run-bash:
	$(BASE_DOCKER_CMD) --interactive --tty $(IMAGE_TAG) /bin/bash

$(VIRTUALENV):
	virtualenv -p $(PYTHON_BIN_PATH) venv

$(SITE_PACKAGES)/cv%:
	@/bin/ln -s `scripts/get_site_package_location.sh`/$(shell basename $@) $@
	@ls $@


