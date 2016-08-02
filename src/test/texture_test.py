import cv2
import aam
import numpy as np
import pytest

from reconstruction.texture import cartesian2barycentric_test


def test_cartesian2barycentric():
    """
    Test if the output of the cartesian2barycentric coordinates are equal to 
    the expected result.
    """
    blue_points = [[20, 20], [50, 160], [160, 20]]
    expected = np.array([
        0.5051020383834839, 0.3571428656578064, 0.13775509595870972
    ])

    x_1 = blue_points[0][0]
    y_1 = blue_points[0][1]
    x_2 = blue_points[1][0]
    y_2 = blue_points[1][1]
    x_3 = blue_points[2][0]
    y_3 = blue_points[2][1]

    x = 50
    y = 70

    lambdas_quick = cartesian2barycentric_test(x_1, y_1, x_2, y_2, x_3, y_3, x, y)

    np.testing.assert_array_equal(lambdas_quick, expected)


@pytest.mark.skipif(True, reason='not suitable for pytest')
def test_sample_from_triangles():
    blue_points = [[20, 20], [50, 160], [160, 20],
                   [50, 20], [60, 200], [180, 20]]

    red_points = [[40, 80], [130, 150], [40, 150],
                  [40, 80], [60, 82], [60, 100]]

    # blue_image = cv2.imread('../data/test_data/blue.png')
    #red_image = cv2.imread('../data/test_data/red.png')
    blue_image = cv2.imread('../data/imm_face_db/01-1m.jpg')
    red_image = cv2.imread('../data/imm_face_db/02-1m.jpg')

    triangles = [[0, 1, 2]]

    #for tri in triangles:
    #    cv2.line(blue_image,
    #             tuple(blue_points[tri[0]]),
    #             tuple(blue_points[tri[1]]), (0, 255, 0), 1)
    #    cv2.line(blue_image,
    #             tuple(blue_points[tri[1]]),
    #             tuple(blue_points[tri[2]]), (0, 255, 0), 1)
    #    cv2.line(blue_image,
    #             tuple(blue_points[tri[2]]),
    #             tuple(blue_points[tri[0]]), (0, 255, 0), 1)

    #for tri in triangles:
    #    cv2.line(red_image,
    #             tuple(red_points[tri[0]]),
    #             tuple(red_points[tri[1]]), (0, 255, 0), 1)
    #    cv2.line(red_image,
    #             tuple(red_points[tri[1]]),
    #             tuple(red_points[tri[2]]), (0, 255, 0), 1)
    #    cv2.line(red_image,
    #             tuple(red_points[tri[2]]),
    #             tuple(red_points[tri[0]]), (0, 255, 0), 1)

    #all_triangles = aam.sample_from_triangles(
    #    red_image, red_points, triangles
    #)

    #cv2.imshow('blue_image', blue_image)
    #cv2.imshow('red_image', red_image)
    #cv2.waitKey(0)
