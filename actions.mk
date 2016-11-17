.PHONY := train_model show_pca test_model show_reconstruction
DEBUG_LEVEL=*

data/imm_face_db: data/imm_face_db.tar.gz
	(cd data; mkdir -p imm_face_db; \
		tar -xvzf imm_face_db.tar.gz -C imm_face_db \
	)

shape_predictor_68_face_landmarks.dat:
	wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -P data/
	(cd data/; bzip2 -d shape_predictor_68_face_landmarks.dat.bz2)

data/imm_face_db.tar.gz:
	(cd data; wget http://www.imm.dtu.dk/~aam/datasets/imm_face_db.tar.gz)

runnit:
	$(BASE_DOCKER_CMD) python main.py


src/reconstruction/texture.so: src/reconstruction/texture.pyx
	$(BASE_DOCKER_CMD) /bin/bash -c '(cd reconstruction; python setup.py build_ext --inplace)'

## IMM Dataset
data/pca_imm_shape_model.npy:
	$(BASE_DOCKER_CMD) python main.py \
		--save_pca_shape \
		--files `./scripts/imm_train_set.sh` \
		--model_shape_file /data/pca_imm_shape_model \
		--shape_type imm

data/pca_imm_texture_model.npy:
	$(BASE_DOCKER_CMD) python main.py \
		--save_pca_texture \
		--files `./scripts/imm_train_set.sh` \
		--model_texture_file /data/pca_imm_texture_model \
		--model_shape_file /data/pca_imm_shape_model.npy \
		--shape_type imm
## END OF IMM

## IBUG using dlib landmark detector
data/pca_ibug_shape_model.npy:
	$(BASE_DOCKER_CMD) python main.py \
		--save_pca_shape \
		--files `./scripts/ibug_train_set.sh` \
		--model_shape_file /data/pca_ibug_shape_model \
		--shape_type ibug

data/pca_ibug_texture_model.npy:
	$(BASE_DOCKER_CMD) python main.py \
		--save_pca_texture \
		--files `./scripts/ibug_train_set.sh` \
		--model_texture_file /data/pca_ibug_texture_model \
		--model_shape_file /data/pca_ibug_shape_model.npy \
		--shape_type ibug
## END OF IBUG


test_model:
	$(BASE_DOCKER_CMD) python main.py \
		--reconstruct \
		--files `./scripts/imm_test_set.sh` \
		--model_texture_file /data/pca_imm_texture_model \
		--model_shape_file /data/pca_shape_model.npy \
		--n_components 6

show_reconstruction:
	$(BASE_DOCKER_CMD) python main.py \
		--reconstruct \
		--files data/imm_face_db/*.asf \
		--model_texture_file /data/pca_imm_texture_model.npy \
		--model_shape_file /data/pca_imm_shape_model.npy \
		--shape_type imm \
		--n_components 6

show_ibug:
	$(BASE_DOCKER_CMD) python main.py \
		--reconstruct \
		--files data/imm_face_db/*.jpg\
		--model_texture_file /data/pca_ibug_texture_model.npy \
		--model_shape_file /data/pca_ibug_shape_model.npy \
		--shape_type ibug

profile_reconstruction:
	$(BASE_DOCKER_CMD) python -m cProfile main.py \
		--reconstruct \
		--files data/imm_face_db/*.asf \
		--model_texture_file /data/pca_imm_texture_model.npy \
		--model_shape_file /data/pca_shape_model.npy \
		--shape_type imm \
		--n_components 6

graph_reconstruction:
	$(BASE_DOCKER_CMD) python main.py \
		--generate_call_graph \
		--files data/imm_face_db/*.asf \
		--model_texture_file /data/pca_imm_texture_model.npy \
		--model_shape_file /data/pca_shape_model.npy \
		--shape_type imm \
		--n_components 6


test_landmarks:
	$(BASE_DOCKER_CMD) python main.py \
		--test_landmarks \
		--image data/test_data/lenna.jpg

.PHONY:= test
test:
	python -m py.test -f src/test/*_test.py

.PHONY:= server
server:
	docker run $(DOCKER_RUN_FLAGS) $(EXTRA_FLAGS) \
		-p 6930:8888 $(IMAGE_TAG) \
		python -m tornado.autoreload server.py

.PHONY:= ember
ember:
	(cd viewer; ember server);

.PHONY:= ctags
ctags:
	ctags --python-kinds=-i src
