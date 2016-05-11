import numpy as np

from aam import get_mean, get_pixel_values
import imm_points
import pca


def test_build_mean_aan():
    imm_points = np.array([
        [[1, 2], [2, 4]],
        [[2, 3], [3, 6]],
    ])

    expected = np.array([
        [1.5, 2.5],
        [2.5, 5.]
    ])

    mean = get_mean(imm_points)

    np.testing.assert_array_equal(mean, expected)


def test_zero_mean_aan():
    imm_points = np.array([
        [1, 2, 2, 4],
        [2, 3, 3, 6],
    ])

    expected = np.array([
        [-0.5, -0.5, -0.5, -1.0],
        [0.5, 0.5, 0.5, 1.0],
    ])

    mean = get_mean(imm_points)
    zero_mean = imm_points - mean

    # test that zero mean has indeed zero mean
    np.testing.assert_array_equal(
        np.mean(zero_mean, axis=0), np.zeros((4))
    )

    np.testing.assert_array_equal(zero_mean, expected)


def test_get_pixel_values():
    asf_file = '../data/imm_face_db/40-2m.asf'
    imm = imm_points.IMMPoints(filename=asf_file)

    points = imm.get_points()
    image = imm.get_image()

    pixels, hull = get_pixel_values(image, points)

    assert False
