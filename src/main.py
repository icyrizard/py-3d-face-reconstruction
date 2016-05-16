import argparse
import logging
import sys

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

    # asf model_files
    pca_group.add_argument(
        '--asf', nargs='+', help='asf files to process'
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
    assert args.asf, '--asf files should be given'
    assert args.model_shape_file, '--model_texture_file needs to be provided to save the pca model'
    assert args.model_texture_file, '--model_texture_file needs to be provided to save the pca model'

    Vt, mean_values, triangles = pca.load(args.model_shape_file)

    textures = aam.build_texture_feature_vector(
        args.asf, imm.get_imm_image_with_landmarks, triangles
    )

    mean_texture = aam.get_mean(textures)
    _, _, Vt = pca.pca(textures, mean_texture)

    pca.save(Vt, mean_texture, triangles, args.model_texture_file)
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
    assert args.asf, '--asf files should be given'
    assert args.model_shape_file, '--model_shape_file needs to be provided to save the pca model'

    points = aam.build_feature_vectors(args.asf,
            imm.get_imm_landmarks, flattened=True)

    mean_values = aam.get_mean(points)

    _, _, Vt = pca.pca(points, mean_values)

    mean_xy = mean_values.reshape((-1, 2))
    triangles = aam.get_triangles(mean_xy[:, 0], mean_xy[:, 1])

    pca.save(Vt, mean_values, triangles, args.model_shape_file)
    logger.info('shape pca model saved in %s', args.model_shape_file + '_shape')


def reconstruct_with_model(args):
    assert args.asf, '--asf files should be given to allow the image to be shown'
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


def main():
    parser = add_parser_options()
    args = parser.parse_args()

    if args.save_pca_shape:
        save_pca_model_shape(args)
    elif args.save_pca_texture:
        save_pca_model_texture(args)
    elif args.reconstruct:
        reconstruct_with_model(args)

if __name__ == '__main__':
    main()
