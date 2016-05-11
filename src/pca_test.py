import numpy as np

from pca import flatten_feature_vectors


def test_flatten_feature_vectors():
    imm_points = np.array([
        [[1, 2], [2, 4]],
        [[2, 3], [3, 6]],
    ])

    expected = np.array([
        [1, 2, 2, 4],
        [2, 3, 3, 6]
    ])

    result = flatten_feature_vectors(imm_points)
    np.testing.assert_array_equal(result, expected)
