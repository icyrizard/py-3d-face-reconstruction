import numpy as np
import cv2
import utils.generate_head_texture as ht
import time

blue_image = cv2.imread('data/test_data/blue.png')
red_image = cv2.imread('data/test_data/red.png')


def test_get_colors_triangle():
    src = blue_image
    dst = red_image

    points2d_src = np.array([
        [20, 20], [50, 160], [160, 20],
        [50, 20], [60, 200], [180, 20]
    ])

    points2d_dst = np.array([
        [40, 80], [130, 150], [40, 150],
        [40, 80], [60, 82], [60, 100]
    ])

    triangles = [[0, 1, 2]]

    t1 = time.clock()

    for i in range(10):
        for t, tri in enumerate(triangles):
            src_p1, src_p2, src_p3 = points2d_src[tri]
            dst_p1, dst_p2, dst_p3 = points2d_dst[tri]

            ht.get_colors_triangle(
                src, dst,
                src_p1[0], src_p1[1],
                src_p2[0], src_p2[1],
                src_p3[0], src_p3[1],
                dst_p1[0], dst_p1[1],
                dst_p2[0], dst_p2[1],
                dst_p3[0], dst_p3[1]
            )

    print (time.clock() - t1) / 10.

    #cv2.imshow('blue_image', blue_image)
    #cv2.imshow('red_image', red_image)
    #cv2.waitKey(0)

    assert False
