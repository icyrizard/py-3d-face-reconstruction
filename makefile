all: data

data: data/imm_face_db

data/imm_face_db: data/imm_face_db.tar.gz
	(cd data; mkdir -p imm_face_db; \
		tar -xvzf imm_face_db.tar.gz -C imm_face_db
	)

data/imm_face_db.tar.gz:
	(cd data; wget http://www.imm.dtu.dk/~aam/datasets/imm_face_db.tar.gz)

generate_test_model:
	python src/main.py \
		--store_pca \
		--asf data/imm_face_db/*.asf \
		--file data/pca_test_model

test:
	py.test -f src/*_test.py

