import cv2
import aam


def test_sample_from_triangles():
    blue_points = [[20, 20], [50, 160], [160, 20],
                   [50, 20], [60, 200], [180, 20]]

    red_points = [[40, 80], [130, 150], [40, 150],
                  [40, 80], [60, 82], [60, 100]]

    # blue_image = cv2.imread('../data/test_data/blue.png')
    #red_image = cv2.imread('../data/test_data/red.png')
    blue_image = cv2.imread('../data/imm_face_db/01-1m.jpg')
    red_image = cv2.imread('../data/imm_face_db/02-1m.jpg')

    triangles = [[0, 1, 2]]

    for tri in triangles:
        cv2.line(blue_image,
                 tuple(blue_points[tri[0]]),
                 tuple(blue_points[tri[1]]), (0, 255, 0), 1)
        cv2.line(blue_image,
                 tuple(blue_points[tri[1]]),
                 tuple(blue_points[tri[2]]), (0, 255, 0), 1)
        cv2.line(blue_image,
                 tuple(blue_points[tri[2]]),
                 tuple(blue_points[tri[0]]), (0, 255, 0), 1)

    for tri in triangles:
        cv2.line(red_image,
                 tuple(red_points[tri[0]]),
                 tuple(red_points[tri[1]]), (0, 255, 0), 1)
        cv2.line(red_image,
                 tuple(red_points[tri[1]]),
                 tuple(red_points[tri[2]]), (0, 255, 0), 1)
        cv2.line(red_image,
                 tuple(red_points[tri[2]]),
                 tuple(red_points[tri[0]]), (0, 255, 0), 1)

    all_triangles = aam.sample_from_triangles(
        red_image, red_points, triangles
    )

    cv2.imshow('blue_image', blue_image)
    cv2.imshow('red_image', red_image)
    cv2.waitKey(0)
