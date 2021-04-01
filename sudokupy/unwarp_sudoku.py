import cv2 as cv
import numpy as np

from detection_utils import in_resize
from gt_annotator import read_ground_truth
import random as rng


def show(img, name='Image'):
    cv.imshow(name, cv.resize(img, (1024, int(img.shape[0] / (img.shape[1] / 1024))), interpolation=cv.INTER_AREA))
    cv.waitKey()


def detect_sudoku(sudoku_img):
    # conversion to gray
    sudoku_gray = cv.cvtColor(sudoku_img, cv.COLOR_BGR2GRAY)

    # Lightness normalization with morphological closing operation (basically subtracts background color)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(25, 25))
    closing = cv.morphologyEx(sudoku_gray, cv.MORPH_CLOSE, kernel)
    sudoku_gray = (sudoku_gray / closing * 255).astype(np.uint8)

    # sudoku_bin = cv.GaussianBlur(sudoku_gray, (5, 5), 0)
    # sudoku_bin = cv.adaptiveThreshold(
    #     sudoku_bin, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, blockSize=21, C=127)

    # Inverse binarization with OTSU to find best threshold automatically (lines should be 1, background 0)
    threshold, sudoku_bin = cv.threshold(sudoku_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # sudoku_bin = cv.medianBlur(sudoku_bin, 5)

    # Dilation to enlarge the binarized structures slightly (fix holes in lines etc.)
    # Must be careful not to over-dilate, otherwise sudoku can merge with surroundings -> bad bbox/contour
    dilation_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    dilated = cv.dilate(sudoku_bin, dilation_kernel)

    # show(dilated)

    def squared_p2p_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy

    image_center = (int(round(sudoku_img.shape[1] / 2)), int(round(sudoku_img.shape[0] / 2)))

    # Finding the largest contour (should be the sudoku)
    contours, hierarchy = cv.findContours(dilated, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_index = -1
    for i in range(len(contours)):
        contour = contours[i]

        # test if the image center is within the proposed region
        polytest = cv.pointPolygonTest(contour, image_center, measureDist=False)
        if polytest < 0:
            continue

        points = cv.boxPoints(cv.minAreaRect(contour))
        a, b, c, d = points
        d1 = squared_p2p_dist(a, b)
        d2 = squared_p2p_dist(a, d)
        # print(d1, ' ', d2)

        square_thresh = (d1 + d2) / 2 * 0.5
        square_diff = abs(d1 - d2)
        if square_diff > square_thresh:
            print(f'Not square {square_diff} {square_thresh}')
            continue

        # test if the contour is large enough
        contour_area = cv.contourArea(contour, oriented=False)
        if contour_area > max_area:
            max_area = contour_area
            max_index = i

    if max_index < 0:
        raise RuntimeError('Sudoku not found.')

    points = cv.boxPoints(cv.minAreaRect(contours[max_index]))
    #     a, b, c, d = points
    #     d1 = squared_p2p_dist(a, b)
    #     d2 = squared_p2p_dist(a, d)
    #     print(d1, ' ', d2, ' ', abs(d1 - d2))

    rect = np.array(points).reshape(4, 2).astype(np.int32)

    # x, y, w, h = cv.boundingRect(contours[max_index])

    # for i in range(len(contours)):
    #     color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    #     cv.drawContours(sudoku_img, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
    # show(sudoku_img, name='contours')

    # img = cv.rectangle(sudoku_img, (x, y), (x + w, y + h), (0, 255, 0), thickness=5)
    # show(img, name='Bounds')

    return rect


annot = read_ground_truth(np.os.path.abspath('ground_truth_new.csv'))
# sudoku_path = annot[0][0]
sudoku_path = annot[18][0]
# sudoku_path = annot[28][0]

for file_path, coords in annot:
    sudoku_img_org = cv.imread(file_path)

    # Scaling such that the longer side is 1024 px long
    input_downscale, sudoku_img = in_resize(sudoku_img_org)

    canvas = sudoku_img.copy()

    image_center = (int(round(sudoku_img.shape[1] / 2)), int(round(sudoku_img.shape[0] / 2)))
    try:
        pred_location = detect_sudoku(sudoku_img)
        # cv.rectangle(canvas, p1, p2, (255, 0, 0), thickness=3)
        cv.polylines(canvas, [pred_location], True, (255, 0, 0), thickness=3)
    except RuntimeError:
        cv.putText(canvas, 'FAILED TO DETECT', image_center, cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3)

    cv.drawMarker(canvas, image_center, (0, 0, 255))

    coords = np.array(coords).reshape(4, 2)
    coords = (coords * input_downscale).astype(np.int32)
    cv.polylines(canvas, [coords], True, (0, 255, 0), thickness=3)

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
