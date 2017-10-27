"""
.. module:: datasets
   :synopsis: Contains ibug dataset abstraction layer


"""
from time import time

import cv2
import numpy as np

import aam
import landmarks
from settings import logger
from .ibug_to_bfm import ibug_mapping

# load detector (this loads the datafile from disk, so needs to be done once).
detector = landmarks.Detector()


class IBUGPoints(aam.AAMPoints):
    SHAPE = (68, 2)

    """IBUG datapoints abstraction"""
    def __init__(self, filename=None, image=None, points_list=None):
        """
        Args:
            filename: optional image file
            points: optional list of x,y points
        """
        assert filename is not None or points_list is not None, 'filename or \
         a ndarray of points list should be given'

        self.filename = filename

        if self.filename:
            if image is None:
                self.__get_image()
            else:
                self.image = image

            points_list = detector.detect_shape(self.image)[0]
            points_list = np.asarray(points_list, dtype=np.float32)

            # normalizing data by dividing it by the image
            points_list[:, 0] /= self.image.shape[1]
            points_list[:, 1] /= self.image.shape[0]

        aam.AAMPoints.__init__(
            self, normalized_flattened_points_list=points_list.flatten(),
            actual_shape=self.SHAPE
        )

    def get_points(self):
        """
        Get the flattened list of points

        Returns:
            ndarray. flattened array of points, see AAMPoints for more
            information.
        """
        return self.normalized_flattened_points_list

    def __get_image(self):
        """
        Get the image corresponding to the self.filename

        Returns:
            ndarray image
        """
        assert hasattr(self, 'filename'), 'filename name should be set, \
                import file must be invoked first'
        self.image = cv2.imread(self.filename)

    def get_image(self):
        """
        Get the image corresponding to the filename
        If filename == image_1.asf, then we read image_1.jpg from disk
        and return this to the user.

        Returns:
            ndarray image
        """
        return self.image

    def show_on_image(self, image, window_name='image', multiply=True):
        self.draw_triangles(image, self.points_list, multiply=multiply)

    def show(self, window_name='image'):
        """show the image and datapoints on the image"""
        assert(len(self.points_list) > 0)
        assert(len(self.filename) > 0)

        image = self.get_image()

        self.draw_triangles(image, self.points_list)


def factory(**kwargs):
    """
    Returns an instance of the dataset aam extending class

    Note that all dataset implementations (in this folder) need to have this
    function to enable transparent use of different datasets throughout this
    project. The reason for this is that we don't want to worry different about
    amounts of landmarks or locations of those landmarks, we just want to use
    them.
    """
    return IBUGPoints(**kwargs)


def get_points(files):
    """
    Args:
        files (array):  Array of images

    Returns:
        ndarray. Array of landmarks.

    """
    points = []
    total_files = len(files)

    for i, filename in enumerate(files):
        t1 = time()
        ibug = IBUGPoints(filename=filename)
        points.append(ibug.get_points())
        logger.debug('processed %s %f, %d/%d', filename, time() - t1, i, total_files)

    return np.asarray(points)


def get_image_with_landmarks(filename):
    """
    Get Points with image and landmarks/points

    Args:
        filename(fullpath): .asf file

    Returns:
        image, points
    """
    ibug = IBUGPoints(filename=filename)

    return ibug.get_image(), ibug.get_points()
