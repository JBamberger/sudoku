import numpy as np
import cv2 as cv


def in_resize(image, long_side=1024):
    h, w = image.shape[:2]

    # scale the longer side to long_side
    scale = long_side / (h if h > w else w)

    return scale, cv.resize(image, None, fx=scale, fy=scale)


def find_sudoku_contour(inverted: np.ndarray) -> np.ndarray:
    dilation_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    dilated = cv.dilate(inverted, dilation_kernel)

    contours, _ = cv.findContours(dilated, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_index = 0
    for i in range(len(contours)):
        a = cv.contourArea(contours[i], oriented=False)
        if a > max_area:
            max_area = a
            max_index = i

    return cv.boundingRect(contours[max_index])
