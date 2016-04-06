import numpy as np


def preprocess(data):
    flattened = []

    y, x, dim = data.shape

    for i in range(y):
        flattened.append(np.ndarray.flatten(data[i]))

    return np.array(flattened)


def pca(data, mean_values, n_components):
    # subtract mean
    zero_mean = data - mean_values
    X = preprocess(zero_mean)

    observations, dims = X.shape

    U, S, V = np.linalg.svd(X)

    return V[:n_components], S[:n_components]
