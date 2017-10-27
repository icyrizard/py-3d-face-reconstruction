import cv2
import numpy as np
import eos

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
    input_points.get_points()

    print dir(eos)

    # scale points to output image shape. We MUST do this.
    points = input_points.get_scaled_points(input_image.shape)
    fit.fit(input_image, points)

    # fit.add(input_image, points)


if __name__ == '__main__':
    fit_model()


## Try seo python bindings
#    model = eos.morphablemodel.load_model(
#        '/usr/local/eos/share/sfm_shape_3448.bin')
#    blend_shapes = eos.morphablemodel.load_blendshapes(
#        '/usr/local/eos/share/expression_blendshapes_3448.bin'
#    )
#
#    s = model.get_shape_model().draw_sample([1.0, -0.5, 0.7, 0.1])
#
#    sample = np.array(s)
#    tri = model.get_shape_model().get_triangle_list()
#    mean = model.get_shape_model().get_mean()
#    dims = model.get_shape_model().get_data_dimension()
#
#    mean = np.array(mean)

#    with open('/data/test.obj', 'w') as f:
#        for i in range(0, len(sample), 3):
#            f.write('v {} {} {}\n'.format(sample[i], sample[i + 1], sample[i + 2])
#        )
#
#        for i in range(0, len(tri)):
#            f.write('f {} {} {}\n'.format(
#                tri[i][0], tri[i][1], tri[i][2],
#            )
#        )

