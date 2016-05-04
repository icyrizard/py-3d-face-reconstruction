import cv2
from imm_points import IMMPoints

state = {}


def cb(index):
    state['index'] = index
    imm_orig = IMMPoints(filename=index)

    state['imm_original'] = imm_orig

    img = cv2.imread('data/imm_face_db/' + imm_orig.filename)
    imm_orig.show_on_img(img)
    cv2.imshow('image', img)
