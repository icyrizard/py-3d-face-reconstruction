from matplotlib.tri import Triangulation
import numpy as np
import cv2

import pca


def get_mean(imm_points):
    """ construct a mean from a matrix of x,y values
    Args:
        imm_points(numpy array) that follows the following structure:

    Returns:
        mean_values (numpy array)

    Examples:
        Input observations:
            0. [[x_0_0, y_0_0], ... , [x_0_m, y_0_m]],
            1. [[x_1_0, y_1_0], ... , [x_1_m, y_1_m]],
            2. [[x_2_0, y_2_0], ... , [x_2_m, y_2_m]],
            3. [[x_3_0, y_3_0], ... , [x_3_m, y_3_m]]

                ....           ....       .....

            n. [[x_4_0, y_4_0], ... , [x_n_m, y_n_m]]

        This vector containts the mean values of the corresponding column, like so:
            0. [[x_0_0, y_0_0], ... , [x_0_k, y_0_k]],
            1. [[x_1_0, y_1_0], ... , [x_1_k, y_1_k]],
            2. [[x_2_0, y_2_0], ... , [x_2_k, y_2_k]],
            3. [[x_3_0, y_3_0], ... , [x_3_k, y_3_k]]

                ....           ....       .....

            n. [[x_4_0, y_4_0], ... , [x_n_k, y_n_k]]

            mean. [[x_mean_0, y_mean_0], ... [x_mean_n, y_mean_n]]
    """
    return np.mean(imm_points, axis=0)


def get_triangles(x_vector, y_vector):
    """ perform triangulation between two 2d vectors"""
    return Triangulation(x_vector, y_vector).triangles


def build_feature_vectors(files, get_points, flattened=False):
    """
    Gets the aam points from the files and appends them seperately to one
    array.

    Args:
        files (list): list files

    return:
        list: list of feature vectors
    """
    points = get_points(files)

    if flattened:
        points = pca.flatten_feature_vectors(points)

    return points


def get_pixel_values(image, points):
    h, w, c = image.shape

    points[:, 0] = points[:, 0] * w
    points[:, 1] = points[:, 1] * h

    image = cv2.blur(image, (3, 3))

    hull = cv2.convexHull(points, returnPoints=True)
    rect = cv2.boundingRect(hull)

    pixels = []
    x, y, w, h = rect

    # pixels = np.zeros((h, w, c), dtype=np.uint8)

    for i in np.linspace(0, 1, num=100):
        for j in np.linspace(0, 1, num=100):
            y_loc_g = int(i * h + y)
            x_loc_g = int(j * w + x)

            y_loc = min(int(i * h), h - 1)
            x_loc = min(int(j * w), w - 1)

            if cv2.pointPolygonTest(hull, (x_loc_g, y_loc_g), measureDist=False) >= 0:
                pixels.extend(image[y_loc_g][x_loc_g])

    return np.asarray(pixels, dtype=np.uint8), hull
