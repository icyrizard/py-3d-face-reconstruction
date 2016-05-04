import copy
import argparse
import logging
import sys

import cv2
import numpy as np

# local imports
import pca

from aam import get_mean
from imm_points import IMMPoints, build_feature_vectors, \
    flatten_feature_vectors

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

        np.load(filename, np.assary([U, s, Vt, mean_values])

        And accessed by:

        UsVtm = np.load(args.model_file)

        U = UsVtm[0]
        s = UsVtm[1]
        Vt = UsVtm[2]
        mean_values = UsVtm[3]

    """
    assert args.asf, '--asf files should be given'
    assert args.model_file, '--model_file needs to be provided to save the pca model'

    imm_points = build_feature_vectors(args.asf, flattened=True)
    mean_values = get_mean(imm_points)

    U, s, Vt = pca.pca(imm_points, mean_values)

    pca.save(U, s, Vt, mean_values, args.model_file)

    logger.info('saved pca model in %s', args.model_file)


def show_pca_model(args):
    assert args.asf, '--asf files should be given to allow the image to be shown'
    assert args.model_file, '--model_file needs to be provided to get the pca model'

    U, s, Vt, mean_values = pca.load(args.model_file)

    # init trackbars
    # index = 0
    # cv2.namedWindow('index')
    # cv2.createTrackbar('index', 'index', index, len(args.asf) - 1, trackbarUpdate)

    n_components = args.n_components
    view.init_eigenvalue_trackbars(n_components, s, window='eigenvalues')

    # use a copy of s to manipulate so we never touch the original
    s_copy = copy.copy(s)
    reconstruction = np.dot(Vt[:n_components], x - mean_values)

    while True:
        imm = IMMPoints(filename=args.asf[index])
        reconstruction = np.dot(V[:n_components], x - mean_values)
        # reconstruction = pca.reconstruct(U[index], s_copy, Vt, n_components, mean_values)

        # reshape to x, y values
        reconstructed = reconstruction.reshape((58, 2))

        imm = IMMPoints(points=reconstructed)
        img = np.full((480, 640, 3), 255, np.uint8)
        imm.show_on_img(img)

        for i in range(n_components):
            s_copy[i] = s[i] + (
                    (cv2.getTrackbarPos(str(i), 'eigenvalues') - 50) / 10.0)

        index = cv2.getTrackbarPos('index', 'index')
        imm.show(window_name='original')

        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

    cv2.destroyAllWindows()


def reconstruct_with_model(args):
    assert args.asf, '--asf files should be given to allow the image to be shown'
    assert args.model_file, '--model_file needs to be provided to get the pca model'

    # clear args. arguments are conflicting with parseargs
    # kivy will parse args upon import and will crash if it finds our
    # 'unsuported by kivy' arguments.
    sys.argv[1:] = []

    from view.reconstruct import ReconstructApp

    U, s, Vt, mean_values = pca.load(args.model_file)
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
