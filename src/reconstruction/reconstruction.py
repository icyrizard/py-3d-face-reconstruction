import cv2
import numpy as np

import pca
import aam
from .texture import fill_triangle_src_dst


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
        cv2.putText(image, str(i), (p[0], p[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (100, 0, 255))
        cv2.circle(image, tuple(p), 3, color=(0, 255, 100))


def get_texture(Points, flattened_texture):
    offset_x, offset_y, w_slice, h_slice = Points.get_bounding_box()
    # Make a rectangle image from the flattened texture array
    return np.asarray(flattened_texture, np.uint8).reshape((h_slice, w_slice, 3))


def scale_eigenvalues(Vt, multiplier_array):
    multipliers = np.ones(Vt.shape[1], dtype=np.float32)
    multipliers[:len(multiplier_array)] = multiplier_array
    Vt = np.dot(np.diag(multipliers), Vt)

    return Vt


def reconstruct_shape(points, shape_model, shape_Vt=None, n_components=None):
    input_points = points.get_points()
    mean_points = shape_model.mean_values
    shape_Vt = shape_Vt if shape_Vt is not None else shape_model.Vt

    reconstructed = pca.reconstruct(
        input_points,
        shape_Vt,
        mean_points,
        n_components=n_components
    )

    points.normalized_flattened_points_list = reconstructed


def reconstruct_texture(src_image, dst_image, texture_model,
        src_points, dst_points, output_points):
    """
    Recontruct texture given the src and dst image

    Args:
        src_image(ndarray): numpy / OpenCV image in BGR
        dst_image(ndarray): numpy image / OpenCVImage, may be None, if None
        then we create an image just as big a the input image but then with a
        black background.
        texture_model(PCAModel): The PCAModel that holds the information that
        we need to reconstruct the image, see pca module.
        Make one by doing this: (see PCAModel on how it is stored in numpy
        file).
        texture_model = pca.PCAModel(model_texture_file)
        src_points(aam.AAMPoints): The AAMPoints object contains the location
        of the landmarks on the face that we need to perform piece wise affine
        warping.
        dst_points(aam.AAMPoints): The AAMPoints object contains the location
        of the landmarks on the face that we need to perform piece wise affine
        warping.
    """
    Vt = texture_model.Vt
    triangles = texture_model.triangles
    mean_texture = texture_model.mean_values

    h, w, c = src_image.shape

    # empty input_texture
    input_texture = np.full((h, w, 3), fill_value=0, dtype=np.uint8)

    points2d_src = src_points.get_scaled_points(src_image.shape)
    points2d_dst = dst_points.get_scaled_points(dst_image.shape)
    points2d_output = output_points.get_scaled_points(src_image.shape)

    # get the texture from the rectangles.
    aam.sample_from_triangles(
        src_image,
        points2d_src,
        points2d_dst,
        triangles,
        input_texture  # this will be filled with the texture afterwards.
    )

    # define the rectangle around the dst_points.
    offset_x, offset_y, w_slice, h_slice = dst_points.get_bounding_box()

    # cut out this region from the input_texture.
    input_texture = input_texture[offset_y: offset_y + h_slice,
                                  offset_x: offset_x + w_slice].flatten()

    # perfrom the PCA reconstruction using the input_texture.
    r_texture = pca.reconstruct(input_texture, Vt, mean_texture)

    # Make an image from the data, the texture is still of type `float`.
    r_texture = np.asarray(r_texture, np.uint8).reshape((h_slice, w_slice, 3))

    # subtract the offset, this is needed because the image is now a
    # small rectangle around the face which starts at [0,0], wheras it first
    # was located at offset_x, offset_y. We need both rectangles to start at
    # [0, 0]. Please note that this should be improved to avoid confusion.
    points2d_dst[:, 0] -= offset_x
    points2d_dst[:, 1] -= offset_y

    points2d_src = points2d_src * 1.1

    # get the texture from the rectangles.
    aam.sample_from_triangles(
        r_texture,
        points2d_dst,  # turn src and dst around
        points2d_output,  # turn src and dst around
        triangles,
        dst_image
    )
