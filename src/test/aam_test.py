import numpy as np
import cv2
import pytest

import aam
import pca
import datasets.imm as imm

from reconstruction import reconstruction


def test_build_mean_aan():
    imm_points = np.array([
        [[1, 2], [2, 4]],
        [[2, 3], [3, 6]],
    ])

    expected = np.array([
        [1.5, 2.5],
        [2.5, 5.]
    ])

    mean = aam.get_mean(imm_points)

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

    mean = aam.get_mean(imm_points)
    zero_mean = imm_points - mean

    # test that zero mean has indeed zero mean
    np.testing.assert_array_equal(
        np.mean(zero_mean, axis=0), np.zeros((4))
    )

    np.testing.assert_array_equal(zero_mean, expected)


def test_build_texture_feature_vectors():
    shape_model = pca.PcaModel('data/test_data/pca_shape_model.npy')
    texture_model = pca.PcaModel('data/test_data/pca_texture_model.npy')

    input_points = imm.IMMPoints(filename='data/imm_face_db/40-3m.asf')
    input_image = input_points.get_image()

    mean_points = imm.IMMPoints(points_list=shape_model.mean_values)
    mean_points = mean_points.get_scaled_points(input_image.shape)
    input_points = input_points.get_scaled_points(input_image.shape)

    assert np.mean(input_points) > 1.0, 'should be greater than 1.0, because \
        it array should be scaled to the image width and height'
    assert np.mean(mean_points) > 1.0, 'should be greater than 1.0, because \
        it array should be scaled to the image width and height'


@pytest.mark.skipif(True, reason='not suitable for pytest')
def test_get_pixel_values():
    asf_file = '../data/imm_face_db/40-2m.asf'
    Vt, s, n_components, mean_shape, triangles = pca.load(args.model_shape_file)

    points = imm.get_points()
    image = imm.get_image()

    pixels, hull = aam.get_pixel_values(image, points)


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

    for tri in triangles:
        cv2.line(blue_image,
                 tuple(blue_points[tri[0]]),
                 tuple(blue_points[tri[1]]), (0, 255, 0), 1)
        cv2.line(blue_image,
                 tuple(blue_points[tri[1]]),
                 tuple(blue_points[tri[2]]), (0, 255, 0), 1)
        cv2.line(blue_image,
                 tuple(blue_points[tri[2]]),
                 tuple(blue_points[tri[0]]), (0, 255, 0), 1)

    for tri in triangles:
        cv2.line(red_image,
                 tuple(red_points[tri[0]]),
                 tuple(red_points[tri[1]]), (0, 255, 0), 1)
        cv2.line(red_image,
                 tuple(red_points[tri[1]]),
                 tuple(red_points[tri[2]]), (0, 255, 0), 1)
        cv2.line(red_image,
                 tuple(red_points[tri[2]]),
                 tuple(red_points[tri[0]]), (0, 255, 0), 1)

    all_triangles = aam.sample_from_triangles(
        red_image, red_points, triangles
    )

    cv2.imshow('blue_image', blue_image)
    cv2.imshow('red_image', red_image)
    cv2.waitKey(0)
