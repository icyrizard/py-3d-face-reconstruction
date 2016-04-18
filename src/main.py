import copy
import argparse
import logging

import cv2
import numpy as np

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
        '--store_pca', action='store_true',
        help='Store the pca model'
    )

    pca_group.add_argument(
        '--show_pca', action='store_true',
        help='Show and manipulate the stored PCA model'
    )

    # asf files
    pca_group.add_argument(
        '--asf', nargs='+', help='asf files to process'
    )

    pca_group.add_argument(
        '--n_components', default=5, type=int,
        help='number of principle components to keep and are able to manipulate'
    )

    pca_group.add_argument(
        '--file', type=str,
        help='pca model file that contains or is going to contain the pca model'
    )

    return parser


def nothing(_):
    pass


def init_eigenvalue_trackbars(n_components, s):
    cv2.namedWindow('eigenvalues')

    for i in range(n_components):
        cv2.createTrackbar('{}'.format(i), 'eigenvalues', 500, 1000, nothing)


def store_pca_model(args):
    """
    Store the U, s, Vt and mean of all the asf datafiles given by the asf
    files.

    It is stored in the following way:

        np.load(filename, np.assary([U, s, Vt, mean_values])

        And accessed by:

        UsVtm = np.load(args.file)

        U = UsVtm[0]
        s = UsVtm[1]
        Vt = UsVtm[2]
        mean_values = UsVtm[3]

    """
    assert args.asf, '--asf files should be given'
    assert args.file, '--file needs to be provided to store the pca model'

    imm_points = build_feature_vectors(args.asf, flattened=True)
    mean_values = get_mean(imm_points)

    U, s, Vt = pca.pca(imm_points, mean_values)

    np.save(args.file, np.asarray([U, s, Vt, mean_values]))
    logger.info('Stored pca model in %s', args.file)


def show_pca_model(args):
    assert args.asf, '--asf files should be given to allow the image to be shown'
    assert args.file, '--file needs to be provided to get the pca model'

    # load the stored model file
    UsVtm = np.load(args.file)

    U = UsVtm[0]
    s = UsVtm[1]
    Vt = UsVtm[2]
    mean_values = UsVtm[3]

    # init trackbars
    index = 0
    cv2.namedWindow('index')
    cv2.createTrackbar('index', 'index', index, len(args.asf) - 1, nothing)

    n_components = args.n_components
    init_eigenvalue_trackbars(n_components, s)

    # use a copy of s to manipulate so we never touch the original
    s_copy = copy.copy(s)

    while True:
        projection = pca.reconstruct(U, s_copy, Vt, n_components)
        X_reconstructed = (projection[index] + mean_values).reshape((58, 2))

        imm = IMMPoints(points=X_reconstructed)
        img = np.full((480, 640, 3), 255, np.uint8)
        imm.show_on_img(img)

        for i in range(n_components):
            s_copy[i] = s[i] * (
                    (cv2.getTrackbarPos(str(i), 'eigenvalues') - 500) / 10.0)

        index = cv2.getTrackbarPos('index', 'index')
        imm = IMMPoints(filename=args.asf[index])
        imm.show(window_name='original')

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()


def main():
    parser = add_parser_options()
    args = parser.parse_args()

    if args.show_pca:
        show_pca_model(args)
    elif args.store_pca:
        store_pca_model(args)
    elif args.reconstruct:
        reconstruct_with_model(args)


if __name__ == '__main__':
    main()
