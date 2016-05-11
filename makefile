all: data

data: data/imm_face_db

data/imm_face_db: data/imm_face_db.tar.gz
	(cd data; mkdir -p imm_face_db; \
		tar -xvzf imm_face_db.tar.gz -C imm_face_db
	)

data/imm_face_db.tar.gz:
	(cd data; wget http://www.imm.dtu.dk/~aam/datasets/imm_face_db.tar.gz)

train_model: data/pca_train_model.npy
	python src/main.py \
		--save_pca \
		--asf data/imm_face_db/*.asf \
		--model_file data/pca_train_model

show_pca:
	python src/main.py \
		--show_pca \
		--asf data/imm_face_db/*.asf \
		--model_file data/pca_model.npy

test_model:
	python src/main.py \
		--reconstruct \
		--asf `./scripts/imm_test_set.sh` \
		--model_file data/pca_train_model.npy \
		--n_components 6

show_reconstruction:
	python src/main.py \
		--reconstruct \
		--asf data/imm_face_db/*.asf \
		--model_file data/pca_train_model.npy \
		--n_components 6

test:
	python -m py.test -f src/*_test.py
