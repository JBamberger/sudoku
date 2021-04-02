import math
import time

import cv2 as cv
import scipy.ndimage
import numpy as np
from detection_utils import in_resize, detect_sudoku, coarse_unwarp, pad_contour, p2p_dist
from gt_annotator import read_ground_truth


def show(img, name='Image', no_wait=False):
    cv.imshow(name, cv.resize(img, (1024, int(img.shape[0] / (img.shape[1] / 1024))), interpolation=cv.INTER_AREA))
    if not no_wait:
        cv.waitKey()


def extract_cells(image):
    image = image.copy()

    sudoku = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # show(sudoku, name='Gray', no_wait=True)

    # erosion_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # sudoku = cv.erode(sudoku, erosion_kernel)
    # show(sudoku, name='Eroded', no_wait=True)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(21, 21))
    closing = cv.morphologyEx(sudoku, cv.MORPH_CLOSE, kernel)
    # show(closing, name='Closing', no_wait=True)

    sudoku = (sudoku / closing * 255).astype(np.uint8)
    # show(sudoku, name='GrayNorm', no_wait=True)

    threshold, sudoku = cv.threshold(sudoku, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # show(sudoku, name='Binarized', no_wait=True)

    sudoku = cv.medianBlur(sudoku, 3)
    # show(sudoku, name='Binarized+Median', no_wait=True)

    dilation_kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    sudoku = cv.dilate(sudoku, dilation_kernel)
    # show(sudoku, name='Dilated', no_wait=True)

    cells = []
    contours, hierarchy = cv.findContours(sudoku, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    contour_canvas = image.copy()
    for i in range(len(contours)):
        # contours[i] = cv.convexHull(contours[i])

        box = np.array(cv.boxPoints(cv.minAreaRect(contours[i]))).astype(np.int32)
        area = cv.contourArea(box)
        if 80 * 80 <= area <= 120 * 120:
            center = box.mean(axis=0)
            cells.append((center, box))
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        contours[i] = box.reshape((-1, 1, 2))

        cv.drawContours(contour_canvas, contours, i, color, 2, cv.LINE_8, hierarchy, 0)

    # compute the cell graph
    nodes = cells
    edges = []
    for i in range(len(nodes)):
        node = nodes[i]
        for j in range(len(nodes)):
            if i == j:
                continue

            other = nodes[j]

            p1 = node[0]
            p2 = other[0]
            dist = p2p_dist(p1, p2)

            e1 = np.array([1, 0])
            diff_norm = (p1 - p2) / dist
            angle = np.arccos(np.dot(e1, diff_norm))

            dd = math.pi / 8

            mod_pi = angle % math.pi
            if mod_pi <= dd or math.pi - dd <= mod_pi:
                direction = 'h'
            elif math.pi / 2 - dd <= mod_pi <= math.pi / 2 + dd:
                direction = 'v'
            else:
                direction = None

            if direction is not None and 80 <= dist <= 115:
                edges.append((i, j, dist, angle, direction))

    # visualization
    for node_num, node in enumerate(nodes):
        center = tuple(node[0].astype(np.int32))
        cv.drawMarker(contour_canvas, center, (0, 255, 0))
        cv.putText(contour_canvas, str(node_num), center, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

    for edge in edges:
        direction = edge[4]
        p1 = tuple(nodes[edge[0]][0].astype(np.int32))
        p2 = tuple(nodes[edge[1]][0].astype(np.int32))

        color = (127, 0, 127) if direction == 'h' else (127, 127, 0)
        cv.line(contour_canvas, p1, p2, color)

    # find the first node
    first_node = 0
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge[0] != first_node:
                continue

            node = nodes[first_node]
            other = nodes[edge[1]]
            orientation = edge[4]
            if orientation == 'h' and node[0][0] > other[0][0] or orientation == 'v' and node[0][1] > other[0][1]:
                first_node = edge[1]
                changed = True
                continue

    cv.drawMarker(contour_canvas, tuple(nodes[first_node][0].astype(np.int32)), (0, 255, 255), markerSize=20,
                  thickness=5)

    edgemap = {i: [edge for edge in filter(lambda e: e[0] == i, edges)] for i in range(len(nodes))}

    cell_to_node = np.zeros((9, 9), dtype=np.int32)
    row_node = col_node = first_node
    for i in range(9):
        for j in range(9):
            cell_to_node[i, j] = col_node

            for edge in edgemap[col_node]:
                if edge[4] == 'h' and nodes[col_node][0][0] < nodes[edge[1]][0][0]:
                    col_node = edge[1]
            # TODO: Detect failures here, i.e. when col_nodes didn't change

        for edge in edgemap[row_node]:
            if edge[4] == 'v' and nodes[row_node][0][1] < nodes[edge[1]][0][1]:
                row_node = edge[1]
                col_node = row_node
        # TODO: Detect failures here, i.e. when row_node didn't change

    print(cell_to_node)

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

        gridpoints = extract_cells(sudoku_coarse_unwarp)

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
