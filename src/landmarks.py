import cv2
import dlib

from settings import LANDMARK_DETECTOR_PATH


def test_detect(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(LANDMARK_DETECTOR_PATH)
    dets = detector(image, 1)

    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        shape = predictor(image, d)

        cv2.rectangle(
            image,
            (d.left(), d.top()),
            (d.right(), d.bottom()),
            [255, 0, 0],
            thickness=2
        )

        for i, p in enumerate(shape.parts()):
            cv2.circle(image, tuple((p.x, p.y)), 3, color=(0, 255, 100))

        cv2.imshow('lenna', image)

    cv2.imwrite('data/out.jpg', image)


class Detector():
    """
    Use dlib library to detect landmarks in a given image
    """
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(LANDMARK_DETECTOR_PATH)

    def detect_faces(self, image):
        """
        Detect faces in an image.

        Args:
            image: np.array

        Returns:
            array of detection arrays.
        """
        # The 1 in the second argument indicates that we should upsample the
        # image 1 time.  This will make everything bigger and allow us to
        # detect more faces.
        return self.detector(image, 1)

    def detect_shape(self, image):
        detections = self.detect_faces(image)
        all_points = []

        for k, d in enumerate(detections):
            points_list = []

            shape = self.predictor(image, d)

            for p in shape.parts():
                points_list.append([p.x, p.y])

            all_points.append(points_list)

        return all_points
