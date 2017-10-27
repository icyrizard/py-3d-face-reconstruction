import cv2

def calc_affine_projection_matrix(landmarks, vertices, image):
    """
    returns the 3x4 camera matrix based on the given locations of the landmarks
    and matching 3d vertices.
    """
    cv2.initCameraMatrix2D(vertices, landmarks, image.shape)
