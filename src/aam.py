import numpy as np


def get_mean(imm_points):
    """ construct a mean from a matrix of x,y values
    Args:
        imm_points(numpy array) that follows the following structure:
    observations:
           0. [[x_0_0, y_0_0], ... , [x_0_m, y_0_m]],
           1. [[x_1_0, y_1_0], ... , [x_1_m, y_1_m]],
           2. [[x_2_0, y_2_0], ... , [x_2_m, y_2_m]],
           3. [[x_3_0, y_3_0], ... , [x_3_m, y_3_m]]

               ....           ....       .m...

           n. [[x_4_0, y_4_0], ... , [x_n_m, y_n_m]]

    Returns mean_values (numpy array)
        This vector containts the mean values of the corresponding column, like so:
           0. [[x_0_0, y_0_0], ... , [x_0_k, y_0_k]],
           1. [[x_1_0, y_1_0], ... , [x_1_k, y_1_k]],
           2. [[x_2_0, y_2_0], ... , [x_2_k, y_2_k]],
           3. [[x_3_0, y_3_0], ... , [x_3_k, y_3_k]]

               ....           ....       .....

           n. [[x_4_0, y_4_0], ... , [x_n_k, y_n_k]]

        mean. [[x_mean_0, y_mean_0], ... [x_mean_n, y_mean_n]]
    """
    return np.mean(imm_points, axis=0)
