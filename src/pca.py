import numpy as np


class PcaModel:
    """
    Abstraction for a pca model file. The pca model is stored in a numpy file
    using numpy.save. The following information is stored:

        Vtm = np.load(model_file)
        self.Vt = Vtm[0]
        self.s = Vtm[1]
        self.n_components = Vtm[2]
        self.mean_values = Vtm[3][0]
        self.triangles = Vtm[4]


    Examples:
        pca = PcaModel(path_to_numpy_model_file)
    """
    def __init__(self, filename=None):
        self.filename = filename

        if filename:
            self.load()

    def save(self):
        """
        Store the information inside this PCA Model instance in a numpy file.

        Args:
            Vt (numpy ndarray): Two dimensional array with dimensions
            s (numpy ndarray): The singular values as a one dimensional array
            n_components: number of components needed to cover .90 percent of the
            variance

        Examples:
            It is stored in the following way:
                np.load(filename, np.assary([Vt, [mean_values]])

            And accessed by:
                Vtm = np.load(args.model_file)

                Vt = Vtm[0]
                mean_values = Vtm[1][0]
                triangles = Vtm[2]

        """
        assert hasattr(self, 'Vt')
        assert hasattr(self, 's')
        assert hasattr(self, 'n_components')
        assert hasattr(self, 'mean_values')
        assert hasattr(self, 'triangles')

        saving = np.asarray(
            [
                self.Vt,
                self.s,
                self.n_components,
                [self.mean_values],
                self.triangles
            ]
        )

        np.save(self.filename, saving)

    def load(self):
        """
        Loads the numpy file, see PcaModel whichs uses this function to load
        the PCA Model data.

        Returns:
            (tuple): Vt, s, n_components, mean_values and triangles
            Vt (numpy ndarray): Two dimensional array with dimensions
            (n_features, n_features)
            n_components: number of components needed to cover .90 percent of the
            variance
            mean_values (numpy ndarray): mean values of the features of the model,
            this should have dimensions (n_featurs, )
            triangles: a list of lists of indices that form a triangles in the
            AAM list.

        Examples:
            We would advise not to use this function directly but to use the
            PcaModel. See the :class:`PcaModel`

        """
        pca_model = np.load(self.filename)
        self.Vt = pca_model[0]
        self.s = pca_model[1]
        self.n_components = pca_model[2]
        self.mean_values = pca_model[3][0]
        self.triangles = pca_model[4]


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
    Store the necessary information for a PCA Model in a numpy file.

    Args:
        Vt (numpy ndarray): Two dimensional array with dimensions
        s (numpy ndarray): The singular values as a one dimensional array
        n_components: number of components needed to cover .90 percent of the
        variance

    Examples:
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


def flatten_feature_vectors(data, dim=0):
    """
    Flattens the feature vectors inside a ndarray

    Args:
        data (numpy array): array of feature vectors
        dim (int): dimension to flatten the data

    Returns:
        array:(numpy array): array flattened feature vectors

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

    """
    flattened = []

    n = data.shape[dim]

    for i in range(n):
        flattened.append(np.ndarray.flatten(data[i]))

    return np.array(flattened)
