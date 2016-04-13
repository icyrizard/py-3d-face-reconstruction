import numpy as np

from aam import build_mean_aam


def test_build_mean_aan():
    imm_points = np.array([
        [[1, 2], [2, 4]],
        [[2, 3], [3, 6]],
    ])

    expected = np.array([
        [1.5, 2.5],
        [2.5, 5.]
    ])

    mean = build_mean_aam(imm_points)

    np.testing.assert_array_equal(mean, expected)


def test_zero_mean_aan():
    imm_points = np.array([
        [[1, 2], [2, 4]],
        [[2, 3], [3, 6]],
    ])

    expected = np.array([
        [[-0.5, -0.5], [-0.5, -1.0]],
        [[0.5, 0.5], [0.5, 1.0]],
    ])

    mean = build_mean_aam(imm_points)
    zero_mean = imm_points - mean

    # test that zero mean has indeed zero mean
    np.testing.assert_array_equal(
        np.mean(zero_mean, axis=0), np.zeros((2, 2))
    )

    np.testing.assert_array_equal(zero_mean, expected)
