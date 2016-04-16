import numpy as np


def pca(data, mean_values):
    # subtract mean
    zero_mean = data - mean_values

    U, s, V = np.linalg.svd(zero_mean, full_matrices=False)

    return U, s, V


def reconstruct(U, s, Vt, n_components):
    return np.dot(U[:, :n_components],
           np.dot(np.diag(s[:n_components]), Vt[:n_components]))
