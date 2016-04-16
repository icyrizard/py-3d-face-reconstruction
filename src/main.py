import copy
import argparse

import cv2
import numpy as np

from pca import pca, reconstruct
from aam import get_mean
from imm_points import IMMPoints, build_feature_vectors, \
    flatten_feature_vectors



def nothing(_):
    pass


def add_parser_options():
    parser = argparse.ArgumentParser(description='IMMPoints tool')

    pca_group = parser.add_argument_group('show_pca')

    pca_group.add_argument(
        '--show_pca', action='store_true',
        help='number of principle components to keep and are able to manipulate'
    )

    # asf files
    pca_group.add_argument(
        '--asf', nargs='+', help='asf files to process'
    )

    pca_group.add_argument(
        '--n_components', default=5, type=int,
        help='number of principle components to keep and are able to manipulate'
    )

    return parser


def init_eigenvalue_trackbars(n_components, s):
    cv2.namedWindow('eigenvalues')

    for i in range(n_components):
        cv2.createTrackbar('{}'.format(i), 'eigenvalues', 500, 1000, nothing)


def show_pca():
    assert args.asf, '--asf files should be supplied'

    imm_points = build_feature_vectors(args.asf, flattened=True)
    mean_values = get_mean(imm_points)

    U, s, Vt = pca(imm_points, mean_values)

    # init trackbars
    index = 0
    cv2.namedWindow('index')
    cv2.createTrackbar('index', 'index', index, len(args.asf) - 1, nothing)

    n_components = args.n_components
    init_eigenvalue_trackbars(n_components, s)

    # use a copy of s to manipulate so we never touch the original
    s_copy = copy.copy(s)

    while True:
        projection = reconstruct(U, s_copy, Vt, n_components)
        X_reconstructed = (projection[index] + mean_values).reshape((58, 2))

        imm = IMMPoints(points=X_reconstructed)
        img = np.full((480, 640, 3), 255, np.uint8)
        imm.show_on_img(img)

        for i in range(n_components):
            s_copy[i] = s[i] * ((cv2.getTrackbarPos(str(i), 'eigenvalues') - 500) / 10.0)

        index = cv2.getTrackbarPos('index', 'index')
        imm = IMMPoints(filename=args.asf[index])
        imm.show(window_name='original')

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = add_parser_options()
    args = parser.parse_args()

    if args.show_pca:
        show_pca()
