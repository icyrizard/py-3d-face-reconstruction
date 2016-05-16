import logging

from matplotlib.tri import Triangulation
import numpy as np
import cv2
import pca

from utils.generate_head_texture import fill_triangle, get_colors_triangle

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def get_mean(vector):
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
    return np.mean(vector, axis=0)


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


def cartesian2barycentric(r1, r2, r3, r):
    x, y = r
    x1, y1 = r1
    x2, y2 = r2
    x3, y3 = r3

    a = np.array([[x1, x2, x3], [y1, y2, y3], [1, 1, 1]])
    b = np.array([x, y, 1])

    return np.linalg.solve(a, b)


def barycentric2cartesian(r1, r2, r3, L):
    x1, y1 = r1
    x2, y2 = r2
    x3, y3 = r3

    a = np.array([[x1, x2, x3], [y1, y2, y3], [1, 1, 1]])
    b = np.array(L)

    return np.asarray(np.dot(a, b), dtype=np.uint32)


def sample_from_triangles(b, points2d_b, triangles, n_samples=20):
    all_triangles = []
    h, w, c = b.shape

    for tri in triangles:
        p1_b = points2d_b[tri[0]]
        p2_b = points2d_b[tri[1]]
        p3_b = points2d_b[tri[2]]

        cv2.line(b,
                 tuple(p1_b),
                 tuple(p2_b), (0, 255, 0), 1)
        cv2.line(b,
                 tuple(p2_b),
                 tuple(p3_b), (0, 255, 0), 1)
        cv2.line(b,
                 tuple(p3_b),
                 tuple(p1_b), (0, 255, 0), 1)

        bary_centric_range = np.linspace(0, 1, num=n_samples)
        pixels = np.full((n_samples * n_samples, 3), fill_value=-1, dtype=np.int)
        L = np.zeros((3, 1))

        for s_i, s in enumerate(bary_centric_range):
            for t_i, t in enumerate(bary_centric_range):
                # make sure the coordinates are inside the triangle
                if s + t <= 1:
                    # build lambda's
                    L[0] = s
                    L[1] = t
                    L[2] = 1 - s - t

                    # cartesian x, y coordinates inside the triangle
                    cart_x, cart_y, _ = barycentric2cartesian(p1_b, p2_b, p3_b, L)
                    pixels[s_i * n_samples + t_i, :] = b[cart_y, cart_x, :]

                    # cv2.circle(b, tuple([cart_x, cart_y]), 1, color=(0, 255, 100))

        all_triangles.append(pixels[np.where(pixels >= 0)])

    return np.asarray(all_triangles, dtype=np.uint8)


def build_texture_feature_vector(files, get_image_with_landmarks, triangles):
    mean_texture = []

    for i, f in enumerate(files[:10]):
        image, landmarks = get_image_with_landmarks(f)
        h, w, c = image.shape
        landmarks[:, 0] = landmarks[:, 0] * w
        landmarks[:, 1] = landmarks[:, 1] * h

        triangles_colors = sample_from_triangles(
            image, landmarks, triangles
        )

        mean_texture.append(triangles_colors)
        logger.info('processed file: {} {}/{}'.format(f, i, len(files)))

        # cv2.imshow('image', image)
        # k = cv2.waitKey(0) & 0xFF

        # if k == 27:
        #     break

    return np.asarray(mean_texture)


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
    for i in np.linspace(0, 1, num=150):
        for j in np.linspace(0, 1, num=150):
            y_loc_g = int(i * h + y)
            x_loc_g = int(j * w + x)

            if cv2.pointPolygonTest(hull, (x_loc_g, y_loc_g), measureDist=False) >= 0:
                image[y_loc_g][x_loc_g][0] = 0
                image[y_loc_g][x_loc_g][1] = 0
                image[y_loc_g][x_loc_g][2] = 0
                pixels.extend(image[y_loc_g][x_loc_g])

    # return np.asarray(pixels, dtype=np.uint8), hull
    return image, hull
