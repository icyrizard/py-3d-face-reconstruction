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
        '--save_pca', action='store_true',
        help='save the pca model'
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
        '--model_file', type=str,
        help='pca model file that contains or is going to contain the pca model'
    )

    return parser


def save_pca_model(args):
    """
    save the U, s, Vt and mean of all the asf datafiles given by the asf
    files.

    It is saved in the following way:

        np.load(filename, np.assary([Vt, mean_values])

        And accessed by:

        Vtm = np.load(args.model_file)

        Vt = Vtm[0]
        mean_values = Vtm[1][0]

    """
    assert args.asf, '--asf files should be given'
    assert args.model_file, '--model_file needs to be provided to save the pca model'

    points = aam.build_feature_vectors(args.asf,
            imm.get_imm_landmarks, flattened=True)
    mean_values = aam.get_mean(points)

    _, _, Vt = pca.pca(points, mean_values)

    pca.save(Vt, mean_values, args.model_file)

    logger.info('saved pca model in %s', args.model_file)


def show_pca_model(args):
    assert args.asf, '--asf files should be given to allow the image to be shown'
    assert args.model_file, '--model_file needs to be provided to get the pca model'

    Vt, mean_values = pca.load(args.model_file)


def reconstruct_with_model(args):
    assert args.asf, '--asf files should be given to allow the image to be shown'
    assert args.model_file, '--model_file needs to be provided to get the pca model'

    # clear sys args. arguments are conflicting with parseargs
    # kivy will parse args upon import and will crash if it finds our
    # 'unsupported by kivy' arguments.
    sys.argv[1:] = []

    from view.reconstruct import ReconstructApp

    Vt, mean_values = pca.load(args.model_file)
    ReconstructApp(
        args=args, eigen_vectors=Vt, mean_values=mean_values,
    ).run()


def main():
    parser = add_parser_options()
    args = parser.parse_args()

    if args.show_pca:
        show_pca_model(args)
    elif args.save_pca:
        save_pca_model(args)
    elif args.reconstruct:
        reconstruct_with_model(args)


if __name__ == '__main__':
    main()
