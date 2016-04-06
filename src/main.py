from pca import pca
import numpy as np
import argparse

from imm_points import IMMPoints


def add_parser_options():
    parser = argparse.ArgumentParser(description='IMMPoints tool')

    # asf files
    parser.add_argument(
        '--asf', nargs='+', help='asf files to process'
    )

    return parser


def build_mean_aam(imm_points):
    """ construct a mean from a matrix of x,y values
    Args:
        imm_points(numpy array)that follows the follwing structure:
    observations:
           0. [[x_0_0, y_0_0], ... , [x_0_m, y_0_m]],
           1. [[x_1_0, y_1_0], ... , [x_1_m, y_1_m]],
           2. [[x_2_0, y_2_0], ... , [x_2_m, y_2_m]],
           3. [[x_3_0, y_3_0], ... , [x_3_m, y_3_m]]

               ....           ....       .m...

           n. [[x_4_0, y_4_0], ... , [x_n_m, y_n_m]]

    Returns mean_values (numpy array)
        This vector containts the mean values of the corresponding column, like so:
           0. [[x_0_0, y_0_0], ... , [x_0_k, y_0_k]],
           1. [[x_1_0, y_1_0], ... , [x_1_k, y_1_k]],
           2. [[x_2_0, y_2_0], ... , [x_2_k, y_2_k]],
           3. [[x_3_0, y_3_0], ... , [x_3_k, y_3_k]]

               ....           ....       .....

           n. [[x_4_0, y_4_0], ... , [x_n_k, y_n_k]]

        mean. [[x_mean_0, y_mean_0], ... [x_mean_n, y_mean_n]]
    """
    mean_values = []

    for i in range(imm_points.shape[1]):
        mean_values.append(np.mean(imm_points[:, i], axis=0))

    return np.array(mean_values)

if __name__ == '__main__':
    parser = add_parser_options()
    args = parser.parse_args()

    if args.asf:
        imm_points = []

        for f in args.asf:
            imm = IMMPoints(filename=f)
            imm_points.append(imm.get_points())
            # imm.show()

        mean_values = build_mean_aam(np.array(imm_points))
        pca(imm_points, mean_values)

        # show immpoints
        imm = IMMPoints(points=mean_values)
        img = np.full((480, 640, 3), 255, np.uint8)
        imm.show_on_img(img)
