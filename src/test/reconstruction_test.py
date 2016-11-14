import cv2

import pca as pca
from settings import logger
from reconstruction import reconstruction

#model_texture_file = '/data/pca_ibug_texture_model.npy'
#model_shape_file = '/data/pca_ibug_shape_model.npy'

model_texture_file = '/data/pca_imm_texture_model.npy'
model_shape_file = '/data/pca_imm_shape_model.npy'

def main():
    shape_components = 58

    shape_model = pca.PCAModel(model_shape_file)
    texture_model = pca.PCAModel(model_texture_file)

    logger.info('using %s shape_components', shape_components)
    image_filename = '/data/imm_face_db/01-1m.asf'

    dst_image = reconstruction.reconstruct_shape_texture(
        'imm',
        shape_model,
        texture_model,
        image_filename,
        shape_components
    )

    cv2.imwrite('/data/reconstructed.png', dst_image)


if __name__ == '__main__':
    main()
