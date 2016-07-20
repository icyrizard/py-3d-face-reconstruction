import numpy as np


class PcaModel:
    """Abstraction for a pca model"""
    def __init__(self, model_file):
        Vtm = np.load(model_file)
        self.Vt = Vtm[0]
        self.s = Vtm[1]
        self.n_components = Vtm[2]
        self.mean_values = Vtm[3][0]
        self.triangles = Vtm[4]


def pca(data, mean_values, variance_percentage=90):
    """
    Perform Singlar Value Decomposition

    Returns:
        U (ndarray): U matrix
        s (ndarray): 1d singular values (diagonal in array form)
        Vt (ndarray): Vt matrix
    """
    # subtract mean
    zero_mean = data - mean_values
    U, s, Vt = np.linalg.svd(zero_mean, full_matrices=False)

    # calculate n_components which captures 90 percent of the variance
    total = s.sum()
    subtotal = 0.0
    i = 0

    while (subtotal * 100.0) / total <= variance_percentage:
        subtotal += s[i]
        i += 1

    n_components = i

    return U, s, Vt, n_components


def reconstruct(feature_vector, Vt, mean_values, n_components=None):
    """
    Reconstruct with U, s, Vt

    Args:
        U (numpy ndarray): One feature vector from the reduced SVD.
            U should have shape (n_features,), (i.e., one dimensional)
        s (numpy ndarray): The singular values as a one dimensional array
        Vt (numpy ndarray): Two dimensional array with dimensions
        (n_features, n_features)
        mean_values (numpy ndarray): mean values of the features of the model,
        this should have dimensions (n_features, )
    """
    if n_components is None:
        n_components = Vt.shape[1]

    zm = feature_vector - mean_values
    yk = np.dot(Vt[:n_components], zm.T)

    return np.dot(Vt[:n_components].T, yk) + mean_values


def save(Vt, s, n_components, mean_values, triangles, filename):
    """
    Store the U, s, Vt and mean of all the asf datafiles given by the asf
    files.

    It is stored in the following way:
        np.load(filename, np.assary([Vt, [mean_values]])

    And accessed by:
        Vtm = np.load(args.model_file)

        Vt = Vtm[0]
        mean_values = Vtm[1][0]
        triangles = Vtm[2]

    """
    saving = np.asarray([Vt, s, n_components, [mean_values], triangles])
    np.save(filename, saving)


def load(filename):
    """
    The model stored by pca.store (see ``pca.store`` method above) is loaded as:
        UsVtm = np.load(args.model_file)

        Vt = Vtm[0]
        mean_values = Vtm[1][0]

        Returns:
           (tuple): Vt, mean_values

            Vt (numpy ndarray): Two dimensional array with dimensions
            (n_features, n_features)
            mean_values (numpy ndarray): mean values of the features of the model,
            this should have dimensions (n_featurs, )
    """
    # load the stored model file
    Vtm = np.load(filename)

    Vt = Vtm[0]
    s = Vtm[1]
    n_components = Vtm[2]
    mean_values = Vtm[3][0]
    triangles = Vtm[4]

    return Vt, s, n_components, mean_values, triangles


#def load_model(filename):
#    # load the stored model file
#    return PcaModel(filename)


def flatten_feature_vectors(data, dim=0):
    """
    Flattens the feature vectors inside a ndarray

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
        dim (int): dimension to flatten the data

    return:
        array: (numpy array): array flattened feature vectors

    """
    flattened = []

    n = data.shape[dim]

    for i in range(n):
        flattened.append(np.ndarray.flatten(data[i]))

    return np.array(flattened)
