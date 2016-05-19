cimport cython
import numpy as np
cimport numpy as np
from cpython cimport array as c_array

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t
DTYPE_float = np.float32
ctypedef np.float32_t  DTYPE_float_t


cdef inline float cross_product(int v1_x, int v1_y, int v2_x, int v2_y):
    # Cross product of two 2d vectors
    return (v1_x * v2_y) - (v1_y * v2_x)

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


def get_colors_triangle(np.ndarray[unsigned char, ndim=3] src,
                        np.ndarray[unsigned char, ndim=3] dest,
                        int src_x1, int src_y1, int src_x2, int src_y2, int
                        src_x3, int src_y3,
                        int dest_x1, int dest_y1, int dest_x2, int dest_y2, int
                        dest_x3, int dest_y3):
    """
    Fill a triangle by applying the Barycentric Algorithm for deciding if a
    point lies inside or outside a triangle.
    """

    # Get the bounding box of the triangle
    cdef int x_min = min(dest_x1, min(dest_x2, dest_x3))
    cdef int x_max = max(dest_x1, max(dest_x2, dest_x3))
    cdef int y_min = min(dest_y1, min(dest_y2, dest_y3))
    cdef int y_max = max(dest_y1, max(dest_y2, dest_y3))

    cdef int vs1_x = dest_x2 - dest_x1
    cdef int vs1_y = dest_y2 - dest_y1
    cdef int vs2_x = dest_x3 - dest_x1
    cdef int vs2_y = dest_y3 - dest_y1

    cdef float s
    cdef float t

    cdef np.ndarray L = np.zeros([3, 1], dtype=DTYPE_float)

    bary_centric_range = np.linspace(0, 1, num=80)

    for s_i, s in enumerate(bary_centric_range):
        for t_i, t in enumerate(bary_centric_range):
            if s + t <= 1:
                a = np.array([
                    [src_x1, src_x2, src_x3],
                    [src_y1, src_y2, src_y3],
                    [1, 1, 1]
                ])

                L[0] = s
                L[1] = t
                L[2] = 1 - s - t

                src_loc = np.dot(a, L)

                a = np.array([
                    [dest_x1, dest_x2, dest_x3],
                    [dest_y1, dest_y2, dest_y3],
                    [1, 1, 1]
                ])

                dest_loc = np.dot(a, L)
                dest[dest_loc[1][0], dest_loc[0][0], :] = src[src_loc[1][0], src_loc[0][0], :]

    # for y in xrange(y_min, y_max):
    #     for x in xrange(x_min, x_max):
    #         q_x = x - dest_x1
    #         q_y = y - dest_y1

    #         s = cross_product(q_x, q_y, vs2_x, vs2_y) / \
    #             cross_product(vs1_x, vs1_y, vs2_x, vs2_y)
    #         t = cross_product(vs1_x, vs1_y, q_x, q_y) / \
    #             cross_product(vs1_x, vs1_y, vs2_x, vs2_y)

    #         if s >= 0 and t >= 0 and s + t <= 1:
    #             a = np.array([
    #                 [src_x1, src_x2, src_x3],
    #                 [src_y1, src_y2, src_y3],
    #                 [1, 1, 1]
    #             ])
    #             L[0] = s
    #             L[1] = t
    #             L[2] = 1 - s - t

    #             src_loc = np.dot(a, L)
    #             dest[y, x, :] = src[src_loc[1][0], src_loc[0][0], :]
