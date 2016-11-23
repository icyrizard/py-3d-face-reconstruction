import cv2

import pca as pca
from settings import logger
from reconstruction import reconstruction
from utility import import_dataset_module

model_texture_file = '/data/pca_ibug_texture_model.npy'
model_shape_file = '/data/pca_ibug_shape_model.npy'

#model_texture_file = '/data/pca_imm_texture_model.npy'
#model_shape_file = '/data/pca_imm_shape_model.npy'


def shape():
    shape_components = 58

    shape_model = pca.PCAModel(model_shape_file)
    texture_model = pca.PCAModel(model_texture_file)

    logger.info('using %s shape_components', shape_components)
    image_filename = '/data/imm_face_db/01-1m.jpg'

    dataset_module = import_dataset_module('ibug')

    dst_image = reconstruction.reconstruct_shape_texture(
        dataset_module,
        shape_model,
        texture_model,
        image_filename,
        shape_components
    )

    cv2.imwrite('/data/reconstructed.png', dst_image)


def fit_model():
    from reconstruction import fit

    shape_components = 58
    shape_model = pca.PCAModel(model_shape_file)
    texture_model = pca.PCAModel(model_texture_file)

    logger.info('using %s shape_components', shape_components)
    image_filename = '/data/imm_face_db/01-1m.jpg'

    dataset_module = import_dataset_module('ibug')

    input_points = dataset_module.factory(filename=image_filename)
    input_image = input_points.get_image()

    fit.fit_model(image)

if __name__ == '__main__':
    #fit_model()
    shape()
