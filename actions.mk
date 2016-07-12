TARGETS += data/pca_shape_train_model.npy
.PHONY := train_model show_pca test_model show_reconstruction

data/imm_face_db: data/imm_face_db.tar.gz
	(cd data; mkdir -p imm_face_db; \
		tar -xvzf imm_face_db.tar.gz -C imm_face_db
	)

train_model:train_shape train_texture
train_texture: data/pca_texture_model.npy
train_shape: data/pca_shape_model.npy

data/imm_face_db.tar.gz:
	(cd data; wget http://www.imm.dtu.dk/~aam/datasets/imm_face_db.tar.gz)

data/pca_shape_model.npy:
	python src/main.py \
		--save_pca_shape \
		--files `./scripts/imm_train_set.sh` \
		--model_shape_file data/pca_shape_model

data/pca_texture_model.npy:
	python src/main.py \
		--save_pca_texture \
		--files `./scripts/imm_train_set.sh` \
		--model_texture_file data/pca_texture_model \
		--model_shape_file data/pca_shape_model.npy

show_pca:
	python src/main.py \
		--show_pca \
		--model_texture_file data/pca_texture_model.npy \
		--model_shape_file data/pca_shape_model.npy


test_model:
	python src/main.py \
		--reconstruct \
		--files `./scripts/imm_test_set.sh` \
		--model_texture_file data/pca_texture_model \
		--model_shape_file data/pca_shape_model.npy \
		--n_components 6

show_reconstruction:
	python src/main.py \
		--reconstruct \
		--files data/imm_face_db/*.asf \
		--model_texture_file data/pca_texture_model.npy \
		--model_shape_file data/pca_shape_model.npy \
		--n_components 6

show_kivy:
	python src/main.py \
		--show_kivy \
		--files data/imm_face_db/*.asf \
		--model_texture_file data/pca_texture_model.npy \
		--model_shape_file data/pca_shape_model.npy \
		--n_components 6

test:
	python -m py.test -f src/*_test.py

test_modules:
	python -m py.test -f src/*/*_test.py

server:
	(cd src/; python -m tornado.autoreload server.py)

ember:
	(cd viewer; ember server);
