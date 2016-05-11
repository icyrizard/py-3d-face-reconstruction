from matplotlib.tri import Triangulation

import cv2
import numpy as np
import argparse
import os


class IMMPoints():
    """Accepts IMM datapoint file which can be shown or used"""
    def __init__(self, filename=None, points=None):
        """
        Args:
            filename: optional .asf file with the imm format
            points: optional list of x,y points
        """
        self.points = points if points is not None else []
        self.filename = filename

        if filename:
            self.import_file(filename)

    def get_points(self):
        return self.points

    def get_image(self):
        cv2.imread(self.image_file)
        return cv2.imread(self.image_file)

    def import_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            data = lines[16:74]
            dir_name = os.path.dirname(filename)
            self.image_file = "{}/{}".format(dir_name, lines[-1].strip())

            for d in data:
                self.points.append(d.split()[2:4])

        self.points = np.asarray(self.points, dtype='f')

    def draw_triangles(self, img, points):
        h, w, c = img.shape

        points[:, 0] = points[:, 0] * w
        points[:, 1] = points[:, 1] * h

        point_indices = list(range(0, 58))
        triangles = Triangulation(points[:, 0], points[:, 1])

        for t, tri in enumerate(triangles.triangles):
            p1, p2, p3 = points[tri]
            cv2.line(img, tuple(p1), tuple(p2), (255, 0, 100), 1)
            cv2.line(img, tuple(p2), tuple(p3), (255, 0, 100), 1)
            cv2.line(img, tuple(p3), tuple(p1), (255, 0, 100), 1)

        for i, p in enumerate(points):
            point_index = int(point_indices[i])
            cv2.putText(img, str(point_index), tuple((p[0], p[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (100, 0, 255))
            cv2.circle(img, tuple(p), 3, color=(0, 255, 100))

    def show_on_img(self, img, window_name='image'):
        self.draw_triangles(img, self.points)

    def show(self, window_name='image'):
        """show the image and datapoints on the image"""
        assert(len(self.points) > 0)
        assert(len(self.filename) > 0)

        img = self.get_image()

        self.draw_triangles(img, self.points)


def get_imm_landmarks(files):
    points = []

    for f in files:
        imm = IMMPoints(filename=f)
        points.append(imm.get_points())

    return np.asarray(points)


def add_parser_options():
    parser = argparse.ArgumentParser(description='IMMPoints tool')

    # asf files
    parser.add_argument(
        'asf', type=str, nargs='+', help='asf files to process'
    )

    return parser

if __name__ == '__main__':
    parser = add_parser_options()
    args = parser.parse_args()

    for f in args.asf:
        imm = IMMPoints(f)
        imm.show()
