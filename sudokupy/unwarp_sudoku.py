import time

import cv2 as cv
import scipy.ndimage
import numpy as np
from detection_utils import in_resize, detect_sudoku, coarse_unwarp, pad_contour
from gt_annotator import read_ground_truth


def show(img, name='Image', no_wait=False):
    cv.imshow(name, cv.resize(img, (1024, int(img.shape[0] / (img.shape[1] / 1024))), interpolation=cv.INTER_AREA))
    if not no_wait:
        cv.waitKey()


def find_gridpoints(image):
    image = image.copy()

    sudoku = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    show(sudoku, name='Gray', no_wait=True)

    # erosion_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # sudoku = cv.erode(sudoku, erosion_kernel)
    # show(sudoku, name='Eroded', no_wait=True)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(21, 21))
    closing = cv.morphologyEx(sudoku, cv.MORPH_CLOSE, kernel)
    show(closing, name='Closing', no_wait=True)

    sudoku = (sudoku / closing * 255).astype(np.uint8)
    show(sudoku, name='GrayNorm', no_wait=True)


    threshold, sudoku = cv.threshold(sudoku, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    show(sudoku, name='Binarized', no_wait=True)

    sudoku = cv.medianBlur(sudoku, 3)
    show(sudoku, name='Binarized+Median', no_wait=True)

    dilation_kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    sudoku = cv.dilate(sudoku, dilation_kernel)
    show(sudoku, name='Dilated', no_wait=True)

    boxes = []
    contours, hierarchy = cv.findContours(sudoku, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    contour_canvas = image.copy()
    for i in range(len(contours)):
        # contours[i] = cv.convexHull(contours[i])

        box = np.array(cv.boxPoints(cv.minAreaRect(contours[i]))).astype(np.int32)
        area = cv.contourArea(box)
        if 7000 <= area <= 13000:
            boxes.append(box)

        contours[i] = box.reshape((-1, 1, 2))

        color = (0, 255, 0) if 7000 <= area <= 13000 else (0, 0, 255)
        cv.drawContours(contour_canvas, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
    show(contour_canvas, name='Contours', no_wait=True)

    return


gt_annoatations = read_ground_truth(np.os.path.abspath('ground_truth_new.csv'))
# sudoku_path = annot[0][0]
# sudoku_path = annot[18][0]
# sudoku_path = annot[28][0]

for file_path, coords in gt_annoatations:
    start = time.time()
    sudoku_img_org = cv.imread(file_path)

    # Scaling such that the longer side is 1024 px long
    input_downscale, sudoku_img = in_resize(sudoku_img_org)

    canvas = sudoku_img.copy()

    image_center = (int(round(sudoku_img.shape[1] / 2)), int(round(sudoku_img.shape[0] / 2)))
    try:
        pred_location = detect_sudoku(sudoku_img)

        # uw = coarse_unwarp(sudoku_img_org, pred_location / input_downscale)
        # show(uw, name='coarse_unwarp')

        # add padding to compensate for bounding boxes that fit too tight
        padded_location = pad_contour(sudoku_img, pred_location)

        # TODO: replace with actual orientation detection
        padded_location = np.roll(padded_location, shift=2, axis=0)

        # unwarp from original image for best warp quality
        sudoku_coarse_unwarp = coarse_unwarp(sudoku_img_org, padded_location / input_downscale)

        gridpoints = find_gridpoints(sudoku_coarse_unwarp)

        # show(sudoku_coarse_unwarp, name='coarse_unwarp')

        # cv.rectangle(canvas, p1, p2, (255, 0, 0), thickness=3)
        cv.polylines(canvas, [pred_location], True, (255, 0, 0), thickness=3)

        for i in range(4):
            cv.putText(canvas, 'ABCD'[i], tuple(pred_location[i, :]),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)



    except RuntimeError:
        cv.putText(canvas, 'FAILED TO DETECT', image_center, cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3)

    cv.drawMarker(canvas, image_center, (0, 0, 255))

    coords = np.array(coords).reshape(4, 2)
    coords = (coords * input_downscale).astype(np.int32)
    cv.polylines(canvas, [coords], True, (0, 255, 0), thickness=3)

    print(time.time() - start)
    show(canvas, name='Bounds')

# sudoku_brightness_normalized

# kernel_size = 25
# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))
#
# closing = cv.morphologyEx(sudoku_gray, cv.MORPH_CLOSE, kernel)
# show(closing)
#
# # opening = cv.morphologyEx(sudoku_gray, cv.MORPH_OPEN, kernel)
# # show(opening)
#
# sudoku_corrected = (sudoku_gray / closing * 255).astype(np.uint8)
# show(sudoku_corrected)
#
# blockSize = 2
# apertureSize = 3
# k = 0.04
#
# dst = cv.cornerHarris(sudoku_corrected, blockSize, apertureSize, k)
#
# dst_norm = np.empty(dst.shape, dtype=np.float32)
# cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
# dst_norm_scaled = cv.convertScaleAbs(dst_norm)
#
# # for i in range(dst_norm.shape[0]):
# #     for j in range(dst_norm.shape[1]):
# #         if int(dst_norm[i, j]) > 100:
# #             cv.circle(dst_norm_scaled, (j, i), 5, (0), 2)
#
# show(dst_norm_scaled)
#
# blurred = cv.GaussianBlur(sudoku_corrected, (5, 5), 0)
# show(blurred)
#
# thresh, sudoku_bin = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# show(sudoku_bin)
