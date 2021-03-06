cimport cython
import numpy as np
cimport numpy as np
from cpython cimport array as c_array
from cython.view cimport array as cvarray

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

DTYPE_float32 = np.float32
DTYPE_float64 = np.float64

ctypedef np.float32_t  DTYPE_float32_t
ctypedef np.float64_t  DTYPE_float64_t
ctypedef float * float_ptr


cdef inline float cross_product(int v1_x, int v1_y, int v2_x, int v2_y):
    # Cross product of two 2d vectors
    return (v1_x * v2_y) - (v1_y * v2_x)


def cartesian2barycentric_test(
        float x1, float y1, float x2, float y2, float x3, float y3, float x, float y):
    """
    lambda_1 = (y_2 - y_3)(x - x_3) + (x_3 - x_2)(y - y_3) /
                (y_2-y_3)(x_1-x_3)+(x_3-x_2)(y_1-y_3)

    lambda_2 = (y_3 - y_1)(x - x_3) + (x_1 - x_3)(y - y_3) /
                (y_2-y_3)(x_1-x_3)+(x_3-x_2)(y_1-y_3)

    lambda_3 = 1 lambda_1 - lambda_2

    Returns:
        ndarray (of dim 3) weights of the barycentric coordinates

    """

    cdef c_array.array dst_loc = c_array.array('f', [0., 0., 0.])

    cartesian2barycentric(
        x1, y1, x2, y2, x3, y3, x, y, dst_loc
    )

    return dst_loc


cdef inline cartesian2barycentric(
        float x_1, float y_1, float x_2, float y_2, float x_3, float y_3, float
        x, float y, float[:] lambdas):
    """
    lambda_1 = (y_2 - y_3)(x - x_3) + (x_3 - x_2)(y - y_3) /
                (y_2-y_3)(x_1-x_3)+(x_3-x_2)(y_1-y_3)
    lambda_2 = (y_3 - y_1)(x - x_3) + (x_1 - x_3)(y - y_3) /
                (y_2-y_3)(x_1-x_3)+(x_3-x_2)(y_1-y_3)
    lambda_3 = 1 lambda_1 - lambda_2

    Returns:
        ndarray (of dim 3) weights of the barycentric coordinates

    """
    cdef float cross_2 = ((y_2 - y_3) * (x_1 - x_3) + (x_3 - x_2) * (y_1 - y_3))

    if (cross_2 <= 0.0):
        cross_2 = 0.01

    lambdas[0] = ((y_2 - y_3) * (x - x_3) + (x_3 - x_2) * (y - y_3)) / cross_2
    lambdas[1] = ((y_3 - y_1) * (x - x_3) + (x_1 - x_3) * (y - y_3)) / cross_2

    lambdas[2] = 1 - lambdas[0] - lambdas[1]


cdef inline np.ndarray[double, ndim=1] cartesian2barycentric_slow(
                int r1_x, r1_y, int r2_x, int r2_y, int r3_x, int r3_y, int
                r_x, int r_y):
    """
    Given a triangle spanned by three cartesion points
    r1, r2, r2, and point r, return the barycentric weights l1, l2, l3.

    Returns:
        ndarray (of dim 3) weights of the barycentric coordinates

    """
    a = np.array([
        [r1_x, r2_x, r3_x],
        [r1_y, r2_y, r3_y],
        [1, 1, 1]
    ])

    b = np.array([r_x, r_y, 1])

    return np.linalg.solve(a, b)


#cdef inline np.ndarray[double, ndim=2] barycentric2cartesian(
#            int x1, int x2, int x3, int y1, int y2, int y3,
#            np.ndarray[long, ndim=2] matrix,
#            np.ndarray[float, ndim=2] Lambdas):
#    matrix[0][0] = x1
#    matrix[0][1] = x2
#    matrix[0][2] = x3
#
#    matrix[1][0] = y1
#    matrix[1][1] = y2
#    matrix[1][2] = y3
#
#    return np.dot(matrix, Lambdas)

cdef inline np.ndarray[double, ndim=2] barycentric2cartesian(
            int x1, int x2, int x3, int y1, int y2, int y3,
            float[:] lambdas, float[:] output):
    # cartesian x
    output[0] = x1 * lambdas[0] + x2 * lambdas[1] + x3 * lambdas[2]

    # cartesian y
    output[1] = y1 * lambdas[0] + y2 * lambdas[1] + y3 * lambdas[2]


@cython.boundscheck(False)
@cython.wraparound(False)
def fill_triangle_src_dst(np.ndarray[unsigned char, ndim=3] src,
                          np.ndarray[unsigned char, ndim=3] dst,
                          int src_x1, int src_y1, int src_x2, int src_y2,
                          int src_x3, int src_y3,
                          int dst_x1, int dst_y1, int dst_x2, int dst_y2,
                          int dst_x3, int dst_y3):
    """
    Fill a triangle by applying the Barycentric Algorithm for deciding if a
    point lies inside or outside a triangle.
    """

    cdef c_array.array dst_loc = c_array.array('f', [0., 0., 0.])
    cdef c_array.array src_loc = c_array.array('f', [0., 0., 0.])

    # get bounding box of the triangle
    cdef np.ndarray triangle_x = np.array([dst_x1, dst_x2, dst_x3])
    cdef np.ndarray triangle_y = np.array([dst_y1, dst_y2, dst_y3])

    cdef int x_min = np.argmin(triangle_x)
    cdef int x_max = np.argmax(triangle_x)
    cdef int y_min = np.argmin(triangle_y)
    cdef int y_max = np.argmax(triangle_y)

    cdef int src_max_dim_x = src.shape[1]
    cdef int src_max_dim_y = src.shape[0]

    cdef int dst_max_dim_x = dst.shape[1]
    cdef int dst_max_dim_y = dst.shape[0]

    # walk over x and y values of this bounding box see if the
    # pixel is in or out the boudning box
    for y in xrange(triangle_y[y_min], triangle_y[y_max]):
        for x in xrange(triangle_x[x_min], triangle_x[x_max]):
            cartesian2barycentric(
                triangle_x[0], triangle_y[0],
                triangle_x[1], triangle_y[1],
                triangle_x[2], triangle_y[2],
                x, y, dst_loc
            )

            s = dst_loc[0]
            t = dst_loc[1]

            # In or out the triangle (with a soft margin)
            if s >= -0.01 and t >= -0.01 and s + t <= 1.01:
                barycentric2cartesian(
                    src_x1, src_x2, src_x3,
                    src_y1, src_y2, src_y3,
                    dst_loc, src_loc
                )

                if src_loc[1] < src_max_dim_y and src_loc[0] < src_max_dim_x \
                        and y < dst_max_dim_y and x < dst_max_dim_x:
                    dst[y, x, :] = src[src_loc[1], src_loc[0], :]


#@cython.boundscheck(False)
#@cython.wraparound(False)
#def fill_triangle(np.ndarray[unsigned char, ndim=3] src,
#                  np.ndarray[unsigned char, ndim=3] dst,
#                  int x1, int y1, int x2, int y2, int x3, int y3):
#    """
#    Fill a triangle by applying the Barycentric Algorithm for deciding if a
#    point lies inside or outside a triangle.
#    """
#
#    # Get the bounding box of the triangle
#    cdef int x_min = min(x1, min(x2, x3))
#    cdef int x_max = max(x1, max(x2, x3))
#    cdef int y_min = min(y1, min(y2, y3))
#    cdef int y_max = max(y1, max(y2, y3))
#
#    cdef int w = x_max - x_min
#    cdef int h = y_max - y_min
#    cdef int new_offset
#
#    cdef c_array.array dst_loc = c_array.array('f', [0., 0., 0.])
#
#    for j, y in enumerate(xrange(y_min, y_max)):
#        for i, x in enumerate(xrange(x_min, x_max)):
#            cartesian2barycentric(
#                x1, y1, x2, y2, x3, y3, x, y, dst_loc
#            )
#
#            s = dst_loc[0]
#            t = dst_loc[1]
#
#            # In or out the triangle (with a soft margin)
#            if s >= -0.01 and t >= -0.01 and s + t <= 1.01:
#                dst[y, x, :] = src[y, x, :]
#
