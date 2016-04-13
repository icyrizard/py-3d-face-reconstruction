import copy
import argparse

import cv2
import numpy as np

from pca import pca, reconstruct
from aam import build_mean_aam
from imm_points import IMMPoints


def nothing(_):
    pass


def add_parser_options():
    parser = argparse.ArgumentParser(description='IMMPoints tool')

    # asf files
    parser.add_argument(
        '--asf', nargs='+', help='asf files to process'
    )

    return parser


def init_eigenvalue_trackbars(n_components, s):
    cv2.namedWindow('eigenvalues')

    for i in range(n_components):
        cv2.createTrackbar('{}'.format(i), 'eigenvalues', 0, 1000, nothing)


if __name__ == '__main__':
    parser = add_parser_options()
    args = parser.parse_args()

    if args.asf:
        imm_points = []

        for f in args.asf:
            imm = IMMPoints(filename=f)
            imm_points.append(imm.get_points())
            # imm.show()

        imm_points = np.array(imm_points)

        mean_values = build_mean_aam(np.array(imm_points))
        U, s, Vt = pca(imm_points, mean_values)

        index = 0
        cv2.namedWindow('index')
        cv2.createTrackbar('index', 'index', index, len(args.asf) - 1, nothing)

        n_components = 5
        init_eigenvalue_trackbars(n_components, s)

        s_copy = copy.copy(s)

        while True:
            projection = reconstruct(U, s_copy, Vt, n_components)
            X_reconstructed = projection[index].reshape((58, 2)) + mean_values

            imm = IMMPoints(points=X_reconstructed)
            img = np.full((480, 640, 3), 255, np.uint8)
            imm.show_on_img(img)

            for i in range(n_components):
                s_copy[i] = s[i] * (cv2.getTrackbarPos(str(i), 'eigenvalues') / 10.0)

            index = cv2.getTrackbarPos('index', 'index')
            imm = IMMPoints(filename=args.asf[index])
            imm.show(window_name='original')

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()
