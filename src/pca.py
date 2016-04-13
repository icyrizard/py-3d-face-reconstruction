import numpy as np


def flatten_feature_vectors(data):
    """
    Flattens the feature vectors inside.

    Example:
        input:
        [
            [[1, 2], [3, 4], [5, 6]],
            ...
            [[1, 2], [3, 4], [5, 6]]
        ]
        output:
        [
            [1, 2, 3, 4, 5, 6],
            ...
            [1, 2, 3, 4, 5, 6]
        ]

    Args:
        data (numpy array): array of feature vectors

    return:
        array: (numpy array): array flattened feature vectors

    """
    flattened = []

    y, x, dim = data.shape

    for i in range(y):
        flattened.append(np.ndarray.flatten(data[i]))

    return np.array(flattened)


def pca(data, mean_values):
    # subtract mean
    zero_mean = data - mean_values
    X = flatten_feature_vectors(zero_mean)

    _, dim = X.shape

    U, s, V = np.linalg.svd(X, full_matrices=False)

    return U, s, V


def reconstruct(U, s, Vt, n_components):
    return np.dot(U[:, :n_components],
        np.dot(np.diag(s[:n_components]), Vt[:n_components]))
