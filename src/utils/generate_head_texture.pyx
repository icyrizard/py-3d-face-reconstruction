cimport cython
import numpy as np
cimport numpy as np
from cpython cimport array as c_array

DTYPE_Int = np.int
ctypedef np.int_t DTYPE_t


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

    y_count = 0
    x_count = 0

    for y in xrange(y_min, y_max):
        for x in xrange(x_min, x_max):
            a = np.ndarray([[x1, x2, x3], [y1, y2, y3], [1, 1, 1]], dtype=DTYPE_Int)
            b = np.ndarray([x, y, 1], dtype=DTYPE_Int)
            L = np.solve(a, b)

            #L[0]

            #if s >= 0 and t >= 0 and s + t <= 1:
            #    dst[y_count * x_max + x_count] = src[y, x, :]
            #    y_count += 1
            #    x_count += 1
