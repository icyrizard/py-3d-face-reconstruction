TARGETS += data/pca_shape_train_model.npy
.PHONY := train_model show_pca test_model show_reconstruction
DEBUG_LEVEL=*

data/imm_face_db: data/imm_face_db.tar.gz
	(cd data; mkdir -p imm_face_db; \
		tar -xvzf imm_face_db.tar.gz -C imm_face_db
	)

train_model: train_shape train_texture
train_texture: data/pca_imm_texture_model.npy data/pca_ibug_texture_model.npy
train_shape: data/pca_imm_shape_model.npy data/pca_ibug_shape_model.npy

data/imm_face_db.tar.gz:
	(cd data; wget http://www.imm.dtu.dk/~aam/datasets/imm_face_db.tar.gz)

runnit:
	$(BASE_DOCKER_CMD) python main.py

compile_texture: src/reconstruction/texture.pyx
	$(BASE_DOCKER_CMD) /bin/bash -c '(cd reconstruction; python setup.py build_ext --inplace)'

data/pca_imm_shape_model.npy:
	$(BASE_DOCKER_CMD) python main.py \
		--save_pca_shape \
		--files `./scripts/imm_train_set.sh` \
		--model_shape_file /data/pca_imm_shape_model \
		--shape_type imm

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

data/pca_imm_texture_model.npy:
	$(BASE_DOCKER_CMD) python main.py \
		--save_pca_texture \
		--files `./scripts/imm_train_set.sh` \
		--model_texture_file /data/pca_imm_texture_model \
		--model_shape_file /data/pca_imm_shape_model.npy \
		--shape_type imm

test_model:
	$(BASE_DOCKER_CMD) python main.py \
		--reconstruct \
		--files `./scripts/imm_test_set.sh` \
		--model_texture_file data/pca_imm_texture_model \
		--model_shape_file data/pca_shape_model.npy \
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
	python src/main.py \
		--reconstruct \
		--files data/imm_face_db/*.jpg\
		--model_texture_file data/pca_ibug_texture_model.npy \
		--model_shape_file data/pca_ibug_shape_model.npy \
		--shape_type ibug

profile_reconstruction:
	python -m cProfile src/main.py \
		--reconstruct \
		--files data/imm_face_db/*.asf \
		--model_texture_file data/pca_imm_texture_model.npy \
		--model_shape_file data/pca_shape_model.npy \
		--shape_type imm \
		--n_components 6

graph_reconstruction:
	python ./src/main.py \
		--generate_call_graph \
		--files data/imm_face_db/*.asf \
		--model_texture_file data/pca_imm_texture_model.npy \
		--model_shape_file data/pca_shape_model.npy \
		--shape_type imm \
		--n_components 6


test_landmarks:
	./src/main.py \
		--test_landmarks \
		--image data/test_data/lenna.jpg

.PHONY:= test
test:
	python -m py.test -f src/test/*_test.py

.PHONY:= server
server:
	#(cd src/; python -m tornado.autoreload server.py)

.PHONY:= ember
ember:
	(cd viewer; ember server);

.PHONY:= ctags
ctags:
	ctags -R --python-kinds=-i src
