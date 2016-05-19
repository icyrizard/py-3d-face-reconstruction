import numpy as np
import cv2

from utils.generate_head_texture import fill_triangle, get_colors_triangle


def cartesian2barycentric(r1, r2, r3, r):
    """
    Given a triangle spanned by three cartesion points
    r1, r2, r2, and point r, return the barycentric weights l1, l2, l3.

    Returns:
        ndarray (of dim 3) weights of the barycentric coordinates

    """
    x, y = r
    x1, y1 = r1
    x2, y2 = r2
    x3, y3 = r3

    a = np.array([[x1, x2, x3], [y1, y2, y3], [1, 1, 1]])
    b = np.array([x, y, 1])

    return np.linalg.solve(a, b)


def barycentric2cartesian(r1, r2, r3, L):
    """
    Given the barycentric weights in L, and cartesian r1, r2, r3 coordinates of
    points that span the triangle, return the cartesian coordinate of the
    points that is located at the weights of L.

    Returns:
        ndarray [x,y] cartesian points.
    """
    x1, y1 = r1
    x2, y2 = r2
    x3, y3 = r3

    a = np.array([[x1, x2, x3], [y1, y2, y3], [1, 1, 1]])
    b = np.array(L)

    return np.asarray(np.dot(a, b), dtype=np.uint32)


def draw_shape(image, points, triangles, multiply=True):
    if multiply:
        h, w, c = image.shape

        points[:, 0] = points[:, 0] * w
        points[:, 1] = points[:, 1] * h

    dim, _ = points.shape

    point_indices = list(range(0, dim))

    for t, tri in enumerate(triangles):
        p1, p2, p3 = points[tri]
        cv2.line(image, tuple(p1), tuple(p2), (255, 0, 100), 1)
        cv2.line(image, tuple(p2), tuple(p3), (255, 0, 100), 1)
        cv2.line(image, tuple(p3), tuple(p1), (255, 0, 100), 1)

    for i, p in enumerate(points):
        point_index = int(point_indices[i])
        cv2.putText(image, str(point_index), (p[0], p[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (100, 0, 255))
        cv2.circle(image, tuple(p), 3, color=(0, 255, 100))


def draw_texture(src, dest, points2d_src, points2d_dest, texture, triangles, multiply=True, n_samples=20):
    texture  = np.asarray(texture, dtype=np.uint8).reshape((-1, 3))

    for t, tri in enumerate(triangles):
        src_p1, src_p2, src_p3 = points2d_src[tri]
        dest_p1, dest_p2, dest_p3 = points2d_dest[tri]

        get_colors_triangle(
            src, dest,
            src_p1[0], src_p1[1],
            src_p2[0], src_p2[1],
            src_p3[0], src_p3[1],
            dest_p1[0], dest_p1[1],
            dest_p2[0], dest_p2[1],
            dest_p3[0], dest_p3[1]
        )
