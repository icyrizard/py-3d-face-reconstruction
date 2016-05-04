import numpy as np


def pca(data, mean_values):
    # subtract mean
    zero_mean = data - mean_values
    U, s, Vt = np.linalg.svd(zero_mean, full_matrices=False)

    return U, s, Vt


def reconstruct(feature_vector, Vt, mean_values, n_components=10):
    """
    Reconstruct with U, s, Vt

    Args:
        U (numpy ndarray): One feature vector from the reduced SVD.
            U should have shape (n_features,), (i.e., one dimensional)
        s (numpy ndarray): The singular values as a one dimensional array
        Vt (numpy ndarray): Two dimensional array with dimensions
        (n_features, n_features)
        mean_values (numpy ndarray): mean values of the features of the model,
        this should have dimensions (n_featurs, )
    """
    zm = feature_vector - mean_values
    yk = np.dot(Vt[:n_components], zm.T)

    return np.dot(Vt[:n_components].T, yk) + mean_values


def save(U, s, Vt, mean_values, filename):
    """
    Store the U, s, Vt and mean of all the asf datafiles given by the asf
    files.

    It is stored in the following way:
        np.load(filename, np.assary([U, s, Vt, mean_values])

    And accessed by:
        UsVtm = np.load(args.model_file)

        U = UsVtm[0]
        s = UsVtm[1]
        Vt = UsVtm[2]
        mean_values = UsVtm[3]

    """
    np.save(filename, np.asarray([U, s, Vt, mean_values]))


def load(filename):
    """
    The model stored by pca.store (see ``pca.store`` method above) is loaded as:
        UsVtm = np.load(args.model_file)

        U = UsVtm[0]
        s = UsVtm[1]
        Vt = UsVtm[2]
        mean_values = UsVtm[3]

        Returns:
           (tuple): U, s, Vt, mean_values

            U (numpy ndarray): One feature vector from the reduced SVD.
                U should have shape (n_features,), (i.e., one dimensional)
            s (numpy ndarray): The singular values as a one dimensional array
            Vt (numpy ndarray): Two dimensional array with dimensions
            (n_features, n_features)
            mean_values (numpy ndarray): mean values of the features of the model,
            this should have dimensions (n_featurs, )
    """
    # load the stored model file
    UsVtm = np.load(filename)

    U = UsVtm[0]
    s = UsVtm[1]
    Vt = UsVtm[2]
    mean_values = UsVtm[3]

    return U, s, Vt, mean_values
