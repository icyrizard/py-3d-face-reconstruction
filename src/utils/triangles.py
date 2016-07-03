import numpy as np
import cv2

from utils.generate_head_texture import fill_triangle, fill_triangle_src_dst

import pca
import aam

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
        #cv2.putText(image, str(point_index), (p[0], p[1]),
        #            cv2.FONT_HERSHEY_SIMPLEX, .5, (100, 0, 255))
        cv2.putText(image, str(i), (p[0], p[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (100, 0, 255))
        cv2.circle(image, tuple(p), 3, color=(0, 255, 100))


def get_texture(Points, flattened_texture):
    offset_x, offset_y, w_slice, h_slice = Points.get_bounding_box()

    # Make a rectangle image from the flattened texture array
    return np.asarray(flattened_texture, np.uint8).reshape((h_slice, w_slice, 3))


def reconstruct_texture(src, dst, Vt, SrcPoints, DstPoints,
                        mean_texture, triangles, n_components):
    # S_mean format
    h, w, c = src.shape
    input_texture = np.full((h, w, 3), fill_value=0, dtype=np.uint8)

    points2d_src = SrcPoints.get_scaled_points(src.shape)
    points2d_dst = DstPoints.get_scaled_points(dst.shape)

    aam.sample_from_triangles(
        src,
        points2d_src,
        points2d_dst,
        triangles,
        input_texture
    )

    offset_x, offset_y, w_slice, h_slice = DstPoints.get_bounding_box()
    input_texture = input_texture[offset_y: offset_y + h_slice,
                                  offset_x: offset_x + w_slice].flatten()

    ## Still in  S_mean format
    r_texture = pca.reconstruct(input_texture, Vt, mean_texture)

    # Make an image from the float data
    r_texture = np.asarray(r_texture, np.uint8).reshape((h_slice, w_slice, 3))

    ## subtract the offset
    points2d_dst[:, 0] -= offset_x
    points2d_dst[:, 1] -= offset_y

    for tri in triangles:
        src_p1, src_p2, src_p3 = points2d_src[tri]
        dst_p1, dst_p2, dst_p3 = points2d_dst[tri]

        fill_triangle_src_dst(
            r_texture, src,
            dst_p1[0], dst_p1[1],
            dst_p2[0], dst_p2[1],
            dst_p3[0], dst_p3[1],
            src_p1[0], src_p1[1],
            src_p2[0], src_p2[1],
            src_p3[0], src_p3[1],
            offset_x,
            offset_y
        )
