cimport cython
import numpy as np
cimport numpy as np
from cpython cimport array as c_array

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

DTYPE_float32 = np.float32
DTYPE_float64 = np.float64

ctypedef np.float32_t  DTYPE_float32_t
ctypedef np.float64_t  DTYPE_float64_t


cdef inline float cross_product(int v1_x, int v1_y, int v2_x, int v2_y):
    # Cross product of two 2d vectors
    return (v1_x * v2_y) - (v1_y * v2_x)


cdef inline np.ndarray[double, ndim=2] barycentric2cartesian(
            int x1, int x2, int x3, int y1, int y2, int y3,
            np.ndarray[long, ndim=2] matrix,
            np.ndarray[float, ndim=2] Lambdas):
    matrix[0][0] = x1
    matrix[0][1] = x2
    matrix[0][2] = x3

    matrix[1][0] = y1
    matrix[1][1] = y2
    matrix[1][2] = y3

    return np.dot(matrix, Lambdas)


@cython.boundscheck(False)
@cython.wraparound(False)
def fill_triangle(np.ndarray[unsigned char, ndim=3] src,
                  np.ndarray[unsigned char, ndim=3] dst,
                  int x1, int y1, int x2, int y2, int x3, int y3):
    """
    Fill a triangle by applying the Barycentric Algorithm for deciding if a
    point lies inside or outside a triangle.
    """

    # Get the bounding box of the triangle
    cdef int x_min = min(x1, min(x2, x3))
    cdef int x_max = max(x1, max(x2, x3))
    cdef int y_min = min(y1, min(y2, y3))
    cdef int y_max = max(y1, max(y2, y3))

    cdef int vs1_x = x2 - x1
    cdef int vs1_y = y2 - y1
    cdef int vs2_x = x3 - x1
    cdef int vs2_y = y3 - y1

    cdef float s
    cdef float t

    for y in xrange(y_min, y_max):
        for x in xrange(x_min, x_max):
            q_x = x - x1
            q_y = y - y1

            s = cross_product(q_x, q_y, vs2_x, vs2_y) / \
                cross_product(vs1_x, vs1_y, vs2_x, vs2_y)
            t = cross_product(vs1_x, vs1_y, q_x, q_y) / \
                cross_product(vs1_x, vs1_y, vs2_x, vs2_y)

            if s >= 0 and t >= 0 and s + t <= 1:
                dst[y, x, :] = src[y, x, :]


@cython.boundscheck(False)
@cython.wraparound(False)
def get_colors_triangle(np.ndarray[unsigned char, ndim=3] src,
                        np.ndarray[unsigned char, ndim=3] dst,
                        int src_x1, int src_y1, int src_x2, int src_y2,
                        int src_x3, int src_y3,
                        int dst_x1, int dst_y1, int dst_x2, int dst_y2,
                        int dst_x3, int dst_y3):
    """
    Fill a triangle by applying the Barycentric Algorithm for deciding if a
    point lies inside or outside a triangle.
    """
    cdef float s
    cdef float t

    cdef np.ndarray L = np.zeros([3, 1], dtype=DTYPE_float32)
    cdef np.ndarray matrix = np.full([3, 3], fill_value=1, dtype=DTYPE_int)

    cdef np.ndarray src_loc = np.zeros([3, 1], dtype=DTYPE_float64)
    cdef np.ndarray dst_loc = np.zeros([3, 1], dtype=DTYPE_float64)

    cdef np.ndarray bary_centric_range = np.linspace(0, 1, num=80)

    # get a float value for every pixel
    for s in bary_centric_range:
        for t in bary_centric_range:
            if s + t <= 1:
                L[0] = s
                L[1] = t
                L[2] = 1 - s - t

                src_loc = barycentric2cartesian(
                    src_x1, src_x2, src_x3,
                    src_y1, src_y2, src_y3,
                    matrix,
                    L
                )

                dst_loc = barycentric2cartesian(
                    dst_x1, dst_x2, dst_x3,
                    dst_y1, dst_y2, dst_y3,
                    matrix,
                    L
                )

                dst[dst_loc[1][0], dst_loc[0][0], :] = src[src_loc[1][0], src_loc[0][0], :]
