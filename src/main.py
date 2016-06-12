# python std
import argparse
import logging
import sys

# installed packages
import cv2
import numpy as np

# local imports
import pca
import aam
import imm_points as imm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def add_parser_options():
    parser = argparse.ArgumentParser(description='IMMPoints tool')
    pca_group = parser.add_argument_group('show_pca')

    pca_group.add_argument(
        '--reconstruct', action='store_true',
        help='Reconstruct one face with a given pca model'
    )

    pca_group.add_argument(
        '--save_pca_shape', action='store_true',
        help='save the pca shape model'
    )

    pca_group.add_argument(
        '--save_pca_texture', action='store_true',
        help='save the pca texture model'
    )

    pca_group.add_argument(
        '--show_pca', action='store_true',
        help='Show and manipulate the saved PCA model'
    )

    pca_group.add_argument(
        '--files', nargs='+', help='files to process'
    )

    pca_group.add_argument(
        '--n_components', default=10, type=int,
        help='number of principle components to keep and are able to manipulate'
    )

    pca_group.add_argument(
        '--model_shape_file', type=str,
        help='pca model file that contains or is going to contain the pca shape model'
    )

    pca_group.add_argument(
        '--model_texture_file', type=str,
        help='pca model file that contains or is going to contain the pca texture model'
    )

    return parser


def save_pca_model_texture(args):
    """
    save the U, s, Vt and mean of all the asf datafiles given by the asf
    files.

    It is saved in the following way:

        np.load(filename, np.assary([Vt, mean_values])

        And accessed by:

        Vtm = np.load(args.model_file_texture)

        Vt = Vtm[0]
        mean_values = Vtm[1][0]

    """
    assert args.files, '--files should be given'
    assert args.model_shape_file, '--model_texture_file needs to be provided to save the pca model'
    assert args.model_texture_file, '--model_texture_file needs to be provided to save the pca model'

    Vt, s, mean_shape, triangles = pca.load(args.model_shape_file)

    textures = aam.build_texture_feature_vectors(
        args.files, imm.get_imm_image_with_landmarks, mean_shape, triangles
    )

    mean_texture = aam.get_mean(textures)
    _, s, Vt = pca.pca(textures, mean_texture)

    pca.save(Vt, s, mean_texture, triangles, args.model_texture_file)

    logger.info('texture pca model saved in %s', args.model_texture_file)


def save_pca_model_shape(args):
    """
    save the U, s, Vt and mean of all the asf datafiles given by the asf
    files.

    It is saved in the following way:

        np.load(filename, np.assary([Vt, mean_values])

        And accessed by:

        Vtm = np.load(args.model_shape_file)

        Vt = Vtm[0]
        mean_values = Vtm[1][0]

    """
    assert args.files, '--files should be given'
    assert args.model_shape_file, '--model_shape_file needs to be provided to save the pca model'

    points = aam.build_shape_feature_vectors(
        args.files, imm.get_imm_points, flattened=True)

    mean_values = aam.get_mean(points)

    _, s, Vt = pca.pca(points, mean_values)

    mean_xy = mean_values.reshape((-1, 2))
    triangles = aam.get_triangles(mean_xy[:, 0], mean_xy[:, 1])

    pca.save(Vt, s, mean_values, triangles, args.model_shape_file)
    logger.info('shape pca model saved in %s', args.model_shape_file + '_shape')


def reconstruct_with_model(args):
    assert args.files, '--files should be given to allow the image to be shown'
    assert args.model_shape_file, '--model_shape_file needs to be provided to get the pca model'

    # clear sys args. arguments are conflicting with parseargs
    # kivy will parse args upon import and will crash if it finds our
    # 'unsupported by kivy' arguments.
    sys.argv[1:] = []

    from view.reconstruct import ReconstructApp
    Vt_shape, mean_values_shape, triangles = pca.load(args.model_shape_file)
    Vt_texture, mean_values_texture, _ = pca.load(args.model_texture_file)

    app = ReconstructApp()

    app.set_values(
        args=args,
        eigenv_shape=Vt_shape,
        eigenv_texture=Vt_texture,
        mean_values_shape=mean_values_shape,
        mean_values_texture=mean_values_texture,
        triangles=triangles
    )

    app.run()


def show_pca_model(args):
    assert args.model_shape_file, '--model_texture_file needs to be provided to save the pca model'
    assert args.model_texture_file, '--model_texture_file needs to be provided to save the pca model'

    from utils.triangles import draw_shape, draw_texture

    Vt_shape, s, mean_values_shape, triangles = pca.load(args.model_shape_file)
    Vt_texture, s_texture, mean_values_texture, _ = pca.load(args.model_texture_file)

    # calculate n_components which captures 90 percent of the variance
    total = s_texture.sum()
    subtotal = 0.0
    i = 0

    while (subtotal * 100.0) / total <= 90.0:
        subtotal += s_texture[i]
        i += 1

    n_components = i

    image = np.full((480, 640, 3), fill_value=255, dtype=np.uint8)

    imm_points = imm.IMMPoints(filename='data/imm_face_db/40-1m.asf')
    input_image = imm_points.get_image()
    input_points = imm_points.get_points()
    h, w, c = input_image.shape

    input_points[:, 0] = input_points[:, 0] * w
    input_points[:, 1] = input_points[:, 1] * h

    mean_values_shape = mean_values_shape.reshape((58, 2))
    mean_values_shape[:, 0] = mean_values_shape[:, 0] * w
    mean_values_shape[:, 1] = mean_values_shape[:, 1] * h

    while True:
        draw_texture(input_image, image, Vt_texture, input_points, mean_values_shape,
                     mean_values_texture, triangles)
        #draw_shape(image, mean_values_shape, triangles, multiply=False)

        cv2.imshow('input_image', input_image)
        cv2.imshow('image', image)
        k = cv2.waitKey(0) & 0xFF

        if k == 27:
            break

    cv2.destroyAllWindows()


def main():
    """main"""
    parser = add_parser_options()
    args = parser.parse_args()

    if args.show_pca:
        show_pca_model(args)
    elif args.save_pca_shape:
        save_pca_model_shape(args)
    elif args.save_pca_texture:
        save_pca_model_texture(args)
    elif args.reconstruct:
        reconstruct_with_model(args)

if __name__ == '__main__':
    main()
